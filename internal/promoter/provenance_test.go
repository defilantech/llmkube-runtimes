package promoter

import (
	"context"
	"errors"
	"strings"
	"testing"
)

func TestVerifyProvenance(t *testing.T) {
	var gotArgs []string
	ok := &Verifier{Repo: "ghcr.io/defilantech/llmkube-llama-vulkan", AttestRepo: "defilantech/llmkube-runtimes",
		run: func(_ context.Context, name string, args ...string) error {
			gotArgs = append([]string{name}, args...)
			return nil
		}}
	if err := ok.Verify(context.Background(), "sha256:abc"); err != nil {
		t.Fatalf("verify: %v", err)
	}
	joined := strings.Join(gotArgs, " ")
	for _, want := range []string{"--type", "https://slsa.dev/provenance/v1", "defilantech/llmkube-runtimes", "sha256:abc"} {
		if !strings.Contains(joined, want) {
			t.Fatalf("args %q missing %q", joined, want)
		}
	}

	bad := &Verifier{run: func(context.Context, string, ...string) error { return errors.New("no signature found") }}
	if err := bad.Verify(context.Background(), "sha256:def"); err == nil {
		t.Fatal("expected verify to fail when cosign errors")
	}
}
