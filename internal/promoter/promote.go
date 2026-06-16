package promoter

import "context"

type Config struct {
	Repo          string
	AttestRepo    string
	StatePath     string
	Namespace     string
	RenderGID     int
	MinDecodeTokS float64
}

func RunOnce(ctx context.Context, cfg Config) error { return nil }
