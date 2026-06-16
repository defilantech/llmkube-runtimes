package promoter

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

type record struct {
	Verdict string `json:"verdict"`
	Tag     string `json:"tag,omitempty"`
}

type State struct {
	Processed map[string]record `json:"processed"` // digest -> record
}

func LoadState(path string) (*State, error) {
	b, err := os.ReadFile(path)
	if os.IsNotExist(err) {
		return &State{Processed: map[string]record{}}, nil
	}
	if err != nil {
		return nil, err
	}
	var s State
	if err := json.Unmarshal(b, &s); err != nil {
		return nil, err
	}
	if s.Processed == nil {
		s.Processed = map[string]record{}
	}
	return &s, nil
}

func (s *State) Seen(digest string) bool { _, ok := s.Processed[digest]; return ok }

func (s *State) Record(digest, verdict, tag string) {
	s.Processed[digest] = record{Verdict: verdict, Tag: tag}
}

// LastRevision returns the highest N already used for tags of the form
// "<ref>-llmkube<N>", or 0 if none.
func (s *State) LastRevision(ref string) int {
	max := 0
	prefix := ref + "-llmkube"
	for _, r := range s.Processed {
		if strings.HasPrefix(r.Tag, prefix) {
			var n int
			if _, err := fmtSscan(r.Tag[len(prefix):], &n); err == nil && n > max {
				max = n
			}
		}
	}
	return max
}

func (s *State) Save(path string) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
		return err
	}
	b, err := json.MarshalIndent(s, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, b, 0o600)
}

func fmtSscan(s string, n *int) (int, error) { return fmt.Sscanf(s, "%d", n) }
