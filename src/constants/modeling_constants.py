from dataclasses import dataclass


@dataclass
class _PipelineSteps:
    oversampling: str = "oversampling"
    imputation: str = "imputation"
    scaler: str = "scaler"
    model: str = "model"


PipelineSteps = _PipelineSteps()


@dataclass
class _ThresholdType:
    decision_function: str = "decision_function"
    predict_proba: str = "predict_proba"


ThresholdType = _ThresholdType()

# Base Pipeline
BASE_PIPELINE_STEPS = [
    (PipelineSteps.oversampling, None),
    (PipelineSteps.imputation, None),
    (PipelineSteps.scaler, None),
    (PipelineSteps.model, None),
]
