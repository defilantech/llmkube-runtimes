package promoter

import (
	"path/filepath"
	"testing"
)

func TestStateRoundTrip(t *testing.T) {
	p := filepath.Join(t.TempDir(), "processed.json")
	s, err := LoadState(p)
	if err != nil {
		t.Fatalf("load empty: %v", err)
	}
	if s.Seen("sha256:abc") {
		t.Fatal("empty state should not have seen abc")
	}
	s.Record("sha256:abc", "promoted", "b9663-llmkube1")
	if err := s.Save(p); err != nil {
		t.Fatalf("save: %v", err)
	}
	s2, err := LoadState(p)
	if err != nil {
		t.Fatalf("reload: %v", err)
	}
	if !s2.Seen("sha256:abc") {
		t.Fatal("reloaded state should have seen abc")
	}
	if got := s2.LastRevision("b9663"); got != 1 {
		t.Fatalf("LastRevision=%d, want 1", got)
	}
}
