"""White-box modules: these read the model's insides. This is the EKG half."""

from __future__ import annotations

from brainscan.contracts import ModuleResult
from brainscan.modules.base import BrainModule, ModuleInfo, RunContext


class NativeCapture(BrainModule):
    """Plain transformers + hidden states. Always available, works on any HF
    causal LM, and is what writes the Scan every UI reads."""

    info = ModuleInfo(
        key="native",
        name="brainscan native capture",
        category="whitebox",
        import_name="transformers",
        pip_extra="default",
        license="Apache-2.0",
        url="https://huggingface.co/docs/transformers",
        what="Residual stream at every layer + logit lens, for any HF model",
    )

    def run(self, ctx: RunContext) -> ModuleResult:
        from brainscan.capture import capture

        if not ctx.model_id:
            raise ValueError("native capture needs ctx.model_id")
        scan = capture(ctx.model_id, ctx.prompt, **ctx.options)
        ctx.scan = scan
        return ModuleResult(
            module=self.info.key,
            category=self.info.category,
            details={"scan_id": scan.meta.scan_id, "n_layers": scan.meta.n_layers},
        )


class TransformerLens(BrainModule):
    info = ModuleInfo(
        key="transformer_lens",
        name="TransformerLens",
        category="whitebox",
        import_name="transformer_lens",
        pip_extra="whitebox",
        license="MIT",
        url="https://github.com/TransformerLensOrg/TransformerLens",
        what="Named hook points on every component; activation patching built in",
    )


class NNSight(BrainModule):
    info = ModuleInfo(
        key="nnsight",
        name="nnsight",
        category="whitebox",
        import_name="nnsight",
        pip_extra="whitebox",
        license="MIT",
        url="https://github.com/ndif-team/nnsight",
        what="Trace + edit any PyTorch model's internals; remote runs on NDIF",
    )
