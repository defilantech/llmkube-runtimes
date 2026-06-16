package promoter

import (
	"os"
	"testing"
)

// The promoter embeds package-local copies of the smoke manifests (assets/),
// because go:embed cannot reach the canonical vulkan/smoke/ directory. These
// guards fail if the embedded copy drifts from the canonical file, which is
// exactly the bug that shipped a stale smoke.sh to the GPU host once.
func TestEmbeddedAssetsMatchCanonical(t *testing.T) {
	cases := []struct {
		name      string
		canonical string
		embedded  string
	}{
		{"smoke.sh", "../../vulkan/smoke/smoke.sh", smokeScript},
		{"job.yaml", "../../vulkan/smoke/job.yaml", jobTemplate},
	}
	for _, c := range cases {
		want, err := os.ReadFile(c.canonical)
		if err != nil {
			t.Fatalf("%s: read canonical: %v", c.name, err)
		}
		if string(want) != c.embedded {
			t.Errorf("%s: embedded copy in internal/promoter/assets/ has drifted from %s; re-copy the canonical file", c.name, c.canonical)
		}
	}
}
