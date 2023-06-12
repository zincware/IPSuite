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
    MACE = "ipsuite.models.MACE"
    Nequip = "ipsuite.models.Nequip"
    Apax = "ipsuite.models.Apax"

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

    # Configuration Comparison
    REMatch = "ipsuite.configuration_comparison.REMatch"
    MMKernel = "ipsuite.configuration_comparison.MMKernel"

    # Configuration Generation
    Packmol = "ipsuite.configuration_generation.Packmol"
    SmilesToAtoms = "ipsuite.configuration_generation.SmilesToAtoms"

    # Data
    AddData = "ipsuite.data_loading.AddData"
    AddDataH5MD = "ipsuite.data_loading.AddDataH5MD"

    # Bootstrap
    RattleAtoms = "ipsuite.bootstrap.RattleAtoms"
    TranslateMolecules = "ipsuite.bootstrap.TranslateMolecules"
    RotateMolecules = "ipsuite.bootstrap.RotateMolecules"

    # Analysis
    DipoleHistogram = "ipsuite.analysis.DipoleHistogram"
    EnergyHistogram = "ipsuite.analysis.EnergyHistogram"
    ForcesHistogram = "ipsuite.analysis.ForcesHistogram"
    ModelEnsembleAnalysis = "ipsuite.analysis.ModelEnsembleAnalysis"
    PredictionMetrics = "ipsuite.analysis.PredictionMetrics"
    ForceAngles = "ipsuite.analysis.ForceAngles"
    RattleAnalysis = "ipsuite.analysis.RattleAnalysis"
    Prediction = "ipsuite.analysis.Prediction"
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

    # calculators
    CP2KSinglePoint = "ipsuite.calculators.CP2KSinglePoint"
    CP2KYaml = "ipsuite.calculators.CP2KYaml"
    ASEGeoOpt = "ipsuite.calculators.ASEGeoOpt"
    ASEMD = "ipsuite.calculators.ASEMD"
    FixedSphereASEMD = "ipsuite.calculators.FixedSphereASEMD"
    xTBSinglePoint = "ipsuite.calculators.xTBSinglePoint"
    LJSinglePoint = "ipsuite.calculators.LJSinglePoint"
    EMTSinglePoint = "ipsuite.calculators.EMTSinglePoint"
    ApaxJaxMD = "ipsuite.calculators.ApaxJaxMD"

    LangevinThermostat = "ipsuite.calculators.LangevinThermostat"
    RescaleBoxModifier = "ipsuite.calculators.RescaleBoxModifier"
    TemperatureRampModifier = "ipsuite.calculators.TemperatureRampModifier"

    # Geometry
    BarycenterMapping = "ipsuite.geometry.BarycenterMapping"


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
