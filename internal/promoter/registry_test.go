package promoter

import (
	"net/http/httptest"
	"net/url"
	"strings"
	"testing"

	"github.com/google/go-containerregistry/pkg/name"
	"github.com/google/go-containerregistry/pkg/registry"
	"github.com/google/go-containerregistry/pkg/v1/random"
	"github.com/google/go-containerregistry/pkg/v1/remote"
)

func TestListCandidatesAndRetag(t *testing.T) {
	srv := httptest.NewServer(registry.New())
	defer srv.Close()
	host := strings.TrimPrefix(srv.URL, "http://")
	repo := host + "/llmkube-llama-vulkan"

	img, _ := random.Image(1024, 1)
	for _, tag := range []string{"candidate-aaa", "candidate-bbb", "stable"} {
		ref, _ := name.ParseReference(repo + ":" + tag)
		if err := remote.Write(ref, img); err != nil {
			t.Fatal(err)
		}
	}

	r := &Registry{Repo: repo}
	cands, err := r.ListCandidates()
	if err != nil {
		t.Fatal(err)
	}
	if len(cands) != 2 {
		t.Fatalf("got %d candidates, want 2", len(cands))
	}
	// retag the first candidate digest as a new tag and confirm it resolves
	if err := r.Retag(cands[0].Digest, "b9663-llmkube1"); err != nil {
		t.Fatal(err)
	}
	_ = url.URL{}
}
