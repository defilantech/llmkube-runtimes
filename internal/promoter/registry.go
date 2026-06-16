package promoter

import (
	"encoding/json"
	"sort"
	"strings"

	"github.com/google/go-containerregistry/pkg/crane"
)

type Candidate struct {
	Tag    string
	Digest string // sha256:...
	Ref    string // upstream llama.cpp ref from the image label, e.g. "b9663"
}

type Registry struct {
	Repo string // e.g. ghcr.io/defilantech/llmkube-llama-vulkan
}

func (r *Registry) ListCandidates() ([]Candidate, error) {
	tags, err := crane.ListTags(r.Repo)
	if err != nil {
		return nil, err
	}
	var out []Candidate
	for _, t := range tags {
		if !strings.HasPrefix(t, "candidate-") {
			continue
		}
		dig, err := crane.Digest(r.Repo + ":" + t)
		if err != nil {
			return nil, err
		}
		out = append(out, Candidate{Tag: t, Digest: dig})
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Tag < out[j].Tag })
	return out, nil
}

// LabelRef reads the io.llmkube.llamacpp.ref OCI label for a digest.
func (r *Registry) LabelRef(digest string) (string, error) {
	cfg, err := crane.Config(r.Repo + "@" + digest)
	if err != nil {
		return "", err
	}
	return extractLabel(cfg, "io.llmkube.llamacpp.ref")
}

// Retag points a new tag at an existing digest without pulling layers.
func (r *Registry) Retag(digest, newTag string) error {
	return crane.Tag(r.Repo+"@"+digest, newTag)
}

func extractLabel(cfgJSON []byte, key string) (string, error) {
	var c struct {
		Config struct {
			Labels map[string]string `json:"Labels"`
		} `json:"config"`
	}
	if err := json.Unmarshal(cfgJSON, &c); err != nil {
		return "", err
	}
	return c.Config.Labels[key], nil
}
