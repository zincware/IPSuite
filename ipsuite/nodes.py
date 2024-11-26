class _Nodes:
    """This class is used to import all the nodes in the package.

    All IPSuite nodes are made available through 'ipsuite.nodes'.
    This avoids nodes with the same name from different modules.
    This allows renaming of modules without changing the 'dvc cmd'
    command, so the stages won't be invalidated.
    """

    # Models
    GAP = "ipsuite.models.GAP"
    EnsembleModel = "ipsuite.models.EnsembleModel"

    # Configuration Selection
    IndexSelection = "ipsuite.configuration_selection.IndexSelection"
    KernelSelection = "ipsuite.configuration_selection.KernelSelection"
    RandomSelection = "ipsuite.configuration_selection.RandomSelection"
    SplitSelection = "ipsuite.configuration_selection.SplitSelection"
    UniformArangeSelection = "ipsuite.configuration_selection.UniformArangeSelection"
    UniformEnergeticSelection = (
        "ipsuite.configuration_selection.UniformEnergeticSelection"
    )
    UniformTemporalSelection = "ipsuite.configuration_selection.UniformTemporalSelection"
    ThresholdSelection = "ipsuite.configuration_selection.ThresholdSelection"
    FilterOutlier = "ipsuite.configuration_selection.FilterOutlier"

    # Configuration Comparison
    REMatch = "ipsuite.configuration_comparison.REMatch"
    MMKernel = "ipsuite.configuration_comparison.MMKernel"

    # Configuration Generation
    Packmol = "ipsuite.configuration_generation.Packmol"
    MultiPackmol = "ipsuite.configuration_generation.MultiPackmol"
    SmilesToAtoms = "ipsuite.configuration_generation.SmilesToAtoms"
    SmilesToConformers = "ipsuite.configuration_generation.SmilesToConformers"
    Smiles2Gromacs = "ipsuite.configuration_generation.Smiles2Gromacs"

    # Data
    AddData = "ipsuite.data_loading.AddData"
    AddDataH5MD = "ipsuite.data_loading.AddDataH5MD"
    ReadData = "ipsuite.data_loading.ReadData"

    # Datasets
    MD22Dataset = "ipsuite.datasets.MD22Dataset"

    # Bootstrap
    RattleAtoms = "ipsuite.bootstrap.RattleAtoms"
    TranslateMolecules = "ipsuite.bootstrap.TranslateMolecules"
    RotateMolecules = "ipsuite.bootstrap.RotateMolecules"
    SurfaceRasterScan = "ipsuite.bootstrap.SurfaceRasterScan"
    SurfaceRasterMetrics = "ipsuite.bootstrap.SurfaceRasterMetrics"

    # Analysis
    DipoleHistogram = "ipsuite.analysis.DipoleHistogram"
    EnergyHistogram = "ipsuite.analysis.EnergyHistogram"
    ForcesHistogram = "ipsuite.analysis.ForcesHistogram"
    StressHistogram = "ipsuite.analysis.StressHistogram"
    ForcesUncertaintyHistogram = "ipsuite.analysis.ForcesUncertaintyHistogram"
    EnergyUncertaintyHistogram = "ipsuite.analysis.EnergyUncertaintyHistogram"
    ModelEnsembleAnalysis = "ipsuite.analysis.ModelEnsembleAnalysis"
    PredictionMetrics = "ipsuite.analysis.PredictionMetrics"
    ForceAngles = "ipsuite.analysis.ForceAngles"
    RattleAnalysis = "ipsuite.analysis.RattleAnalysis"
    Prediction = "ipsuite.analysis.Prediction"
    CalibrationMetrics = "ipsuite.analysis.CalibrationMetrics"
    BoxScale = "ipsuite.analysis.BoxScale"
    BoxHeatUp = "ipsuite.analysis.BoxHeatUp"
    NaNCheck = "ipsuite.analysis.NaNCheck"
    ConnectivityCheck = "ipsuite.analysis.ConnectivityCheck"
    EnergySpikeCheck = "ipsuite.analysis.EnergySpikeCheck"
    MDStability = "ipsuite.analysis.MDStability"
    MoveSingleParticle = "ipsuite.analysis.MoveSingleParticle"
    AnalyseGlobalForceSensitivity = "ipsuite.analysis.AnalyseGlobalForceSensitivity"
    AnalyseSingleForceSensitivity = "ipsuite.analysis.AnalyseSingleForceSensitivity"
    ForceDecomposition = "ipsuite.analysis.ForceDecomposition"
    ThresholdCheck = "ipsuite.analysis.ThresholdCheck"
    TemperatureCheck = "ipsuite.analysis.TemperatureCheck"
    FixedSphereConstraint = "ipsuite.calculators.FixedSphereConstraint"
    FixedLayerConstraint = "ipsuite.calculators.FixedLayerConstraint"
    AnalyseDensity = "ipsuite.analysis.AnalyseDensity"

    # calculators
    CP2KSinglePoint = "ipsuite.calculators.CP2KSinglePoint"
    CP2KYaml = "ipsuite.calculators.CP2KYaml"
    ASEGeoOpt = "ipsuite.calculators.ASEGeoOpt"
    ASEMD = "ipsuite.calculators.ASEMD"
    GPAWSinglePoint = "ipsuite.calculators.GPAWSinglePoint"
    xTBSinglePoint = "ipsuite.calculators.xTBSinglePoint"
    LJSinglePoint = "ipsuite.calculators.LJSinglePoint"
    EMTSinglePoint = "ipsuite.calculators.EMTSinglePoint"
    OrcaSinglePoint = "ipsuite.calculators.OrcaSinglePoint"
    LammpsSimulator = "ipsuite.calculators.LammpsSimulator"
    MixCalculator = "ipsuite.calculators.MixCalculator"

    LangevinThermostat = "ipsuite.calculators.LangevinThermostat"
    VelocityVerletDynamic = "ipsuite.calculators.VelocityVerletDynamic"
    NPTThermostat = "ipsuite.calculators.NPTThermostat"
    SVCRBarostat = "ipsuite.calculators.SVCRBarostat"
    RescaleBoxModifier = "ipsuite.calculators.RescaleBoxModifier"
    BoxOscillatingRampModifier = "ipsuite.calculators.BoxOscillatingRampModifier"
    TemperatureRampModifier = "ipsuite.calculators.TemperatureRampModifier"
    TemperatureOscillatingRampModifier = (
        "ipsuite.calculators.TemperatureOscillatingRampModifier"
    )
    TemperatureRampModifier = "ipsuite.calculators.TemperatureRampModifier"

    # Geometry
    BarycenterMapping = "ipsuite.geometry.BarycenterMapping"

    # Data manipulation
    Flatten = "ipsuite.base.Flatten"


def __getattr__(name):
    """Overwrite the default __getattr__ to import the nodes lazily."""
    import importlib

    _name = getattr(_Nodes, name)

    module, class_name = _name.rsplit(".", 1)
    module = importlib.import_module(module)
    return getattr(module, class_name)


def __dir__() -> list:
    """Return a list of all the nodes in the package."""
    return [name for name in dir(_Nodes) if not name.startswith("_")]


__all__ = __dir__()
