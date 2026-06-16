package promoter

import (
	"context"
	"path/filepath"
	"testing"
)

type fakeReg struct {
	cands   []Candidate
	labels  map[string]string
	retags  []string // "digest=>tag"
	listErr error
}

func (f *fakeReg) ListCandidates() ([]Candidate, error) { return f.cands, f.listErr }
func (f *fakeReg) LabelRef(digest string) (string, error) {
	return f.labels[digest], nil
}
func (f *fakeReg) Retag(digest, newTag string) error {
	f.retags = append(f.retags, digest+"=>"+newTag)
	return nil
}

type fakeVerifier struct {
	verified []string
	fail     map[string]bool
}

func (f *fakeVerifier) Verify(_ context.Context, digest string) error {
	f.verified = append(f.verified, digest)
	if f.fail[digest] {
		return errContext("no signature")
	}
	return nil
}

type errContext string

func (e errContext) Error() string { return string(e) }

type fakeSmoke struct {
	ran  []string
	fail map[string]bool
}

func (f *fakeSmoke) RunFor(_ context.Context, image string, _ int, _ float64) (bool, error) {
	f.ran = append(f.ran, image)
	if f.fail[image] {
		return false, nil
	}
	return true, nil
}

func TestRunOncePipeline(t *testing.T) {
	statePath := filepath.Join(t.TempDir(), "processed.json")

	// Pre-seed: candidate-old already processed.
	pre, _ := LoadState(statePath)
	pre.Record("sha256:old", "promoted", "b9663-llmkube1")
	if err := pre.Save(statePath); err != nil {
		t.Fatal(err)
	}

	reg := &fakeReg{
		cands: []Candidate{
			{Tag: "candidate-old", Digest: "sha256:old"},
			{Tag: "candidate-new", Digest: "sha256:new"},
		},
		labels: map[string]string{"sha256:new": "b9663"},
	}
	ver := &fakeVerifier{fail: map[string]bool{}}
	smk := &fakeSmoke{fail: map[string]bool{}}

	p := &Pipeline{
		Repo:          "ghcr.io/defilantech/llmkube-llama-vulkan",
		StatePath:     statePath,
		RenderGID:     110,
		MinDecodeTokS: 40,
		reg:           reg,
		verifier:      ver,
		smoker:        smk,
	}
	if err := p.Run(context.Background()); err != nil {
		t.Fatalf("Run: %v", err)
	}

	// Only the new digest is verified.
	if len(ver.verified) != 1 || ver.verified[0] != "sha256:new" {
		t.Fatalf("verified=%v, want [sha256:new]", ver.verified)
	}
	// Only the new image is smoked.
	if len(smk.ran) != 1 {
		t.Fatalf("smoked=%v, want 1", smk.ran)
	}
	// Retags: stable + b9663-llmkube2 (since llmkube1 already in state).
	want := map[string]bool{
		"sha256:new=>stable":         true,
		"sha256:new=>b9663-llmkube2": true,
	}
	if len(reg.retags) != 2 {
		t.Fatalf("retags=%v, want 2", reg.retags)
	}
	for _, r := range reg.retags {
		if !want[r] {
			t.Fatalf("unexpected retag %q (retags=%v)", r, reg.retags)
		}
	}

	// State now records both digests.
	post, _ := LoadState(statePath)
	if !post.Seen("sha256:old") || !post.Seen("sha256:new") {
		t.Fatalf("state missing digests: %+v", post)
	}
	if post.Processed["sha256:new"].Verdict != "promoted" {
		t.Fatalf("new verdict=%q, want promoted", post.Processed["sha256:new"].Verdict)
	}
	if post.Processed["sha256:new"].Tag != "b9663-llmkube2" {
		t.Fatalf("new tag=%q, want b9663-llmkube2", post.Processed["sha256:new"].Tag)
	}
}

func TestRunOncePipelineFailedProvenance(t *testing.T) {
	statePath := filepath.Join(t.TempDir(), "processed.json")
	reg := &fakeReg{
		cands:  []Candidate{{Tag: "candidate-bad", Digest: "sha256:bad"}},
		labels: map[string]string{"sha256:bad": "b9663"},
	}
	ver := &fakeVerifier{fail: map[string]bool{"sha256:bad": true}}
	smk := &fakeSmoke{fail: map[string]bool{}}
	p := &Pipeline{
		Repo: "ghcr.io/x", StatePath: statePath, MinDecodeTokS: 40,
		reg: reg, verifier: ver, smoker: smk,
	}
	if err := p.Run(context.Background()); err != nil {
		t.Fatalf("Run: %v", err)
	}
	if len(smk.ran) != 0 {
		t.Fatalf("smoke ran on bad-provenance candidate: %v", smk.ran)
	}
	if len(reg.retags) != 0 {
		t.Fatalf("retagged a bad-provenance candidate: %v", reg.retags)
	}
	post, _ := LoadState(statePath)
	if post.Processed["sha256:bad"].Verdict != "failed-provenance" {
		t.Fatalf("verdict=%q, want failed-provenance", post.Processed["sha256:bad"].Verdict)
	}
}
