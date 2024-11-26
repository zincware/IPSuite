from .version import __version__

# Base imports
from .base import Flatten

# Models
from .models import GAP, EnsembleModel

# Configuration Selection
from .configuration_selection import (
    IndexSelection,
    KernelSelection,
    RandomSelection,
    SplitSelection,
    UniformArangeSelection,
    UniformEnergeticSelection,
    UniformTemporalSelection,
    ThresholdSelection,
    FilterOutlier,
)

# Configuration Comparison
from .configuration_comparison import REMatch, MMKernel

# Configuration Generation
from .configuration_generation import (
    Packmol,
    MultiPackmol,
    SmilesToAtoms,
    SmilesToConformers,
    Smiles2Gromacs,
)

# Data
from .data_loading import AddData, AddDataH5MD, ReadData

# Datasets
from .datasets import MD22Dataset

# Bootstrap
from .bootstrap import (
    RattleAtoms,
    TranslateMolecules,
    RotateMolecules,
    SurfaceRasterScan,
    SurfaceRasterMetrics,
)

# Analysis
from .analysis import (
    DipoleHistogram,
    EnergyHistogram,
    ForcesHistogram,
    StressHistogram,
    ForcesUncertaintyHistogram,
    EnergyUncertaintyHistogram,
    ModelEnsembleAnalysis,
    PredictionMetrics,
    ForceAngles,
    RattleAnalysis,
    Prediction,
    CalibrationMetrics,
    BoxScale,
    BoxHeatUp,
    NaNCheck,
    ConnectivityCheck,
    EnergySpikeCheck,
    MDStability,
    MoveSingleParticle,
    AnalyseGlobalForceSensitivity,
    AnalyseSingleForceSensitivity,
    ForceDecomposition,
    ThresholdCheck,
    TemperatureCheck,
    AnalyseDensity,
)

# Calculators
from .calculators import (
    CP2KSinglePoint,
    CP2KYaml,
    ASEGeoOpt,
    ASEMD,
    xTBSinglePoint,
    LJSinglePoint,
    EMTSinglePoint,
    OrcaSinglePoint,
    LammpsSimulator,
    MixCalculator,
    LangevinThermostat,
    VelocityVerletDynamic,
    NPTThermostat,
    SVCRBarostat,
    RescaleBoxModifier,
    BoxOscillatingRampModifier,
    TemperatureRampModifier,
    TemperatureOscillatingRampModifier,
)

# Geometry
from .geometry import BarycenterMapping

# Project
from .project import Project

# Update __all__ for lazy loading
__all__ = [
    "__version__",
    # Base
    "Flatten",
    # Models
    "GAP",
    "EnsembleModel",
    # Configuration Selection
    "IndexSelection",
    "KernelSelection",
    "RandomSelection",
    "SplitSelection",
    "UniformArangeSelection",
    "UniformEnergeticSelection",
    "UniformTemporalSelection",
    "ThresholdSelection",
    "FilterOutlier",
    # Configuration Comparison
    "REMatch",
    "MMKernel",
    # Configuration Generation
    "Packmol",
    "MultiPackmol",
    "SmilesToAtoms",
    "SmilesToConformers",
    "Smiles2Gromacs",
    # Data
    "AddData",
    "AddDataH5MD",
    "ReadData",
    # Datasets
    "MD22Dataset",
    # Bootstrap
    "RattleAtoms",
    "TranslateMolecules",
    "RotateMolecules",
    "SurfaceRasterScan",
    "SurfaceRasterMetrics",
    # Analysis
    "DipoleHistogram",
    "EnergyHistogram",
    "ForcesHistogram",
    "StressHistogram",
    "ForcesUncertaintyHistogram",
    "EnergyUncertaintyHistogram",
    "ModelEnsembleAnalysis",
    "PredictionMetrics",
    "ForceAngles",
    "RattleAnalysis",
    "Prediction",
    "CalibrationMetrics",
    "BoxScale",
    "BoxHeatUp",
    "NaNCheck",
    "ConnectivityCheck",
    "EnergySpikeCheck",
    "MDStability",
    "MoveSingleParticle",
    "AnalyseGlobalForceSensitivity",
    "AnalyseSingleForceSensitivity",
    "ForceDecomposition",
    "ThresholdCheck",
    "TemperatureCheck",
    "AnalyseDensity",
    # Calculators
    "CP2KSinglePoint",
    "CP2KYaml",
    "ASEGeoOpt",
    "ASEMD",
    "xTBSinglePoint",
    "LJSinglePoint",
    "EMTSinglePoint",
    "OrcaSinglePoint",
    "LammpsSimulator",
    "MixCalculator",
    "LangevinThermostat",
    "VelocityVerletDynamic",
    "NPTThermostat",
    "SVCRBarostat",
    "RescaleBoxModifier",
    "BoxOscillatingRampModifier",
    "TemperatureRampModifier",
    "TemperatureOscillatingRampModifier",
    # Geometry
    "BarycenterMapping",
    # Project
    "Project",
]
