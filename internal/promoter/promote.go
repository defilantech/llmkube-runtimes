package promoter

import (
	"context"
	"fmt"
	"log/slog"
)

type Config struct {
	Repo          string
	AttestRepo    string
	StatePath     string
	Namespace     string
	RenderGID     int
	MinDecodeTokS float64
}

// Narrow seams so the pipeline is trivially fakeable in tests.
type candidateRegistry interface {
	ListCandidates() ([]Candidate, error)
	LabelRef(digest string) (string, error)
	Retag(digest, newTag string) error
}

type provenanceVerifier interface {
	Verify(ctx context.Context, digest string) error
}

type smoker interface {
	RunFor(ctx context.Context, image string, renderGID int, floorTokS float64) (bool, error)
}

type Pipeline struct {
	Repo          string
	Namespace     string
	StatePath     string
	RenderGID     int
	MinDecodeTokS float64

	reg      candidateRegistry
	verifier provenanceVerifier
	smoker   smoker
}

// RunFor adapts the real Smoke runner to the smoker seam.
func (s *Smoke) RunFor(ctx context.Context, image string, renderGID int, floorTokS float64) (bool, error) {
	s.Image = image
	s.RenderGID = renderGID
	s.FloorTokS = floorTokS
	if err := s.EnsureScriptConfigMap(ctx); err != nil {
		return false, err
	}
	return s.Run(ctx)
}

func RunOnce(ctx context.Context, cfg Config) error {
	p := &Pipeline{
		Repo:          cfg.Repo,
		Namespace:     cfg.Namespace,
		StatePath:     cfg.StatePath,
		RenderGID:     cfg.RenderGID,
		MinDecodeTokS: cfg.MinDecodeTokS,
		reg:           &Registry{Repo: cfg.Repo},
		verifier:      NewVerifier(cfg.Repo, cfg.AttestRepo),
		smoker:        &Smoke{Namespace: cfg.Namespace},
	}
	return p.Run(ctx)
}

func (p *Pipeline) Run(ctx context.Context) error {
	state, err := LoadState(p.StatePath)
	if err != nil {
		return fmt.Errorf("load state: %w", err)
	}

	cands, err := p.reg.ListCandidates()
	if err != nil {
		return fmt.Errorf("list candidates: %w", err)
	}

	for _, c := range cands {
		if state.Seen(c.Digest) {
			slog.Debug("skip already-processed candidate", "tag", c.Tag, "digest", c.Digest)
			continue
		}
		p.process(ctx, state, c)
		if err := state.Save(p.StatePath); err != nil {
			return fmt.Errorf("save state: %w", err)
		}
	}
	return nil
}

// process runs a single candidate through verify -> smoke -> promote, recording
// its terminal verdict in state. Halt-and-continue: any failure records and
// returns; it never aborts the whole run.
func (p *Pipeline) process(ctx context.Context, state *State, c Candidate) {
	image := p.Repo + "@" + c.Digest

	if err := p.verifier.Verify(ctx, c.Digest); err != nil {
		// Should never happen for a real CI build, so it is suspicious.
		slog.Warn("provenance verification failed", "tag", c.Tag, "digest", c.Digest, "err", err)
		state.Record(c.Digest, "failed-provenance", "")
		return
	}

	ref, err := p.reg.LabelRef(c.Digest)
	if err != nil {
		slog.Error("read image label", "tag", c.Tag, "digest", c.Digest, "err", err)
		state.Record(c.Digest, "failed-label", "")
		return
	}
	if ref == "" {
		slog.Error("image missing io.llmkube.llamacpp.ref label", "tag", c.Tag, "digest", c.Digest)
		state.Record(c.Digest, "failed-label", "")
		return
	}

	passed, err := p.smoker.RunFor(ctx, image, p.RenderGID, p.MinDecodeTokS)
	if err != nil {
		slog.Error("smoke run errored", "tag", c.Tag, "digest", c.Digest, "err", err)
		state.Record(c.Digest, "failed-smoke", "")
		return
	}
	if !passed {
		slog.Info("smoke failed", "tag", c.Tag, "digest", c.Digest)
		state.Record(c.Digest, "failed-smoke", "")
		return
	}

	n := state.LastRevision(ref) + 1
	revTag := fmt.Sprintf("%s-llmkube%d", ref, n)
	for _, tag := range []string{"stable", revTag} {
		if err := p.reg.Retag(c.Digest, tag); err != nil {
			slog.Error("retag failed", "digest", c.Digest, "tag", tag, "err", err)
			state.Record(c.Digest, "failed-retag", "")
			return
		}
	}
	slog.Info("promoted candidate", "tag", c.Tag, "digest", c.Digest, "revision", revTag)
	state.Record(c.Digest, "promoted", revTag)
}
