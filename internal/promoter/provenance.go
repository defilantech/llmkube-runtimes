package promoter

import (
	"context"
	"fmt"
	"os/exec"
)

type Verifier struct {
	Repo       string
	AttestRepo string
	run        func(ctx context.Context, name string, args ...string) error
}

func NewVerifier(repo, attestRepo string) *Verifier {
	return &Verifier{Repo: repo, AttestRepo: attestRepo, run: execRun}
}

func execRun(ctx context.Context, name string, args ...string) error {
	cmd := exec.CommandContext(ctx, name, args...)
	out, err := cmd.CombinedOutput()
	if err != nil {
		return fmt.Errorf("%s: %v: %s", name, err, out)
	}
	return nil
}

// Verify confirms the digest carries a build-provenance attestation produced by
// the AttestRepo's GitHub Actions workflow (keyless OIDC). Fails closed.
func (v *Verifier) Verify(ctx context.Context, digest string) error {
	run := v.run
	if run == nil {
		run = execRun
	}
	ref := v.Repo + "@" + digest
	return run(ctx, "cosign", "verify-attestation",
		"--type", "slsaprovenance",
		"--certificate-oidc-issuer", "https://token.actions.githubusercontent.com",
		"--certificate-identity-regexp", fmt.Sprintf("^https://github.com/%s/", v.AttestRepo),
		ref,
	)
}
