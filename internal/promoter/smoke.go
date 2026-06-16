package promoter

import (
	"context"
	_ "embed"
	"fmt"
	"os/exec"
	"strconv"
	"strings"
)

// The smoke manifests are embedded from a package-local copy under assets/.
// Go's embed directive cannot reference parent directories (../../vulkan/smoke),
// so the canonical files in vulkan/smoke/ are mirrored into assets/ and kept in
// sync. This keeps the promoter binary self-contained.
//
//go:embed assets/job.yaml
var jobTemplate string

//go:embed assets/smoke.sh
var smokeScript string

type Smoke struct {
	Image     string
	RenderGID int
	FloorTokS float64
	Namespace string
	apply     func(ctx context.Context, manifest string) (jobName string, err error)
	wait      func(ctx context.Context, jobName string) (passed bool, logs string, err error)
}

func (s *Smoke) render() string {
	r := strings.NewReplacer(
		"__IMAGE__", s.Image,
		"__RENDER_GID__", strconv.Itoa(s.RenderGID),
		"__FLOOR__", strconv.FormatFloat(s.FloorTokS, 'f', -1, 64),
	)
	return r.Replace(jobTemplate)
}

func (s *Smoke) namespace() string {
	if s.Namespace == "" {
		return "llmkube-promoter"
	}
	return s.Namespace
}

func (s *Smoke) Run(ctx context.Context) (bool, error) {
	apply, wait := s.apply, s.wait
	if apply == nil {
		apply = s.kubectlApply
	}
	if wait == nil {
		wait = s.kubectlWaitJob
	}
	job, err := apply(ctx, s.render())
	if err != nil {
		return false, fmt.Errorf("apply smoke job: %w", err)
	}
	passed, logs, err := wait(ctx, job)
	if err != nil {
		return false, fmt.Errorf("wait smoke job %s: %w", job, err)
	}
	if !passed {
		return false, fmt.Errorf("smoke failed for %s:\n%s", s.Image, logs)
	}
	return true, nil
}

// EnsureScriptConfigMap creates/updates the vk-smoke-script ConfigMap from the
// embedded smoke.sh so the Job can mount it. Idempotent via apply.
func (s *Smoke) EnsureScriptConfigMap(ctx context.Context) error {
	ns := s.namespace()
	cmd := exec.CommandContext(ctx, "kubectl", "-n", ns,
		"create", "configmap", "vk-smoke-script",
		"--from-literal=smoke.sh="+smokeScript,
		"--dry-run=client", "-o", "yaml")
	manifest, err := cmd.Output()
	if err != nil {
		return fmt.Errorf("render smoke configmap: %w", err)
	}
	return kubectlApplyStdin(ctx, ns, string(manifest))
}

func (s *Smoke) kubectlApply(ctx context.Context, manifest string) (string, error) {
	ns := s.namespace()
	cmd := exec.CommandContext(ctx, "kubectl", "-n", ns, "create", "-f", "-", "-o", "name")
	cmd.Stdin = strings.NewReader(manifest)
	out, err := cmd.CombinedOutput()
	if err != nil {
		return "", fmt.Errorf("kubectl create: %v: %s", err, out)
	}
	// out is like "job.batch/vk-smoke-xyz"
	name := strings.TrimSpace(string(out))
	if i := strings.LastIndex(name, "/"); i >= 0 {
		name = name[i+1:]
	}
	return name, nil
}

func (s *Smoke) kubectlWaitJob(ctx context.Context, jobName string) (bool, string, error) {
	ns := s.namespace()
	// Wait for either completion or failure; ignore the wait exit code and read
	// the terminal condition explicitly afterwards.
	_ = exec.CommandContext(ctx, "kubectl", "-n", ns, "wait",
		"--for=condition=complete", "--timeout=300s", "job/"+jobName).Run()

	logsOut, _ := exec.CommandContext(ctx, "kubectl", "-n", ns,
		"logs", "job/"+jobName).CombinedOutput()
	logs := string(logsOut)

	statusOut, err := exec.CommandContext(ctx, "kubectl", "-n", ns,
		"get", "job/"+jobName, "-o", "jsonpath={.status.succeeded}").Output()
	if err != nil {
		return false, logs, fmt.Errorf("kubectl get job status: %w", err)
	}
	succeeded := strings.TrimSpace(string(statusOut)) == "1"
	return succeeded, logs, nil
}

func kubectlApplyStdin(ctx context.Context, ns, manifest string) error {
	cmd := exec.CommandContext(ctx, "kubectl", "-n", ns, "apply", "-f", "-")
	cmd.Stdin = strings.NewReader(manifest)
	out, err := cmd.CombinedOutput()
	if err != nil {
		return fmt.Errorf("kubectl apply: %v: %s", err, out)
	}
	return nil
}
