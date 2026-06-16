package promoter

import (
	"context"
	"strings"
	"testing"
)

func TestSmokeRunnerTemplatesAndParses(t *testing.T) {
	var applied string
	s := &Smoke{
		Image: "ghcr.io/x/llmkube-llama-vulkan@sha256:abc", RenderGID: 110, FloorTokS: 40,
		apply: func(_ context.Context, manifest string) (string, error) { applied = manifest; return "vk-smoke-xyz", nil },
		wait:  func(_ context.Context, job string) (bool, string, error) { return true, "PASS\n", nil },
	}
	ok, err := s.Run(context.Background())
	if err != nil || !ok {
		t.Fatalf("Run ok=%v err=%v", ok, err)
	}
	if !strings.Contains(applied, "sha256:abc") || !strings.Contains(applied, "supplementalGroups: [110]") {
		t.Fatalf("manifest not templated: %s", applied)
	}
}
