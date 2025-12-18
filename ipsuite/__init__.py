"""The ipsuite package."""

# <AUTOGEN_INIT>
from ipsuite.utils.helpers import setup_ase
from ipsuite.utils.logs import setup_logging

setup_logging(__name__)
setup_ase()

# fmt: off
# <AUTOGEN_INIT>
import lazy_loader


__getattr__, __dir__, __all__ = lazy_loader.attach_stub(__name__, __file__)

__all__ = ['ASEMD', 'ASEMDSafeSampling', 'ATOMS_LST', 'AddData', 'AddDataH5MD',
           'AllowedStructuresFilter', 'AnalyseAtoms', 'AnalyseDensity',
           'AnalyseGlobalForceSensitivity', 'AnalyseSingleForceSensitivity',
           'AnalyseStructureMeanForce', 'ApplyCalculator', 'Atoms',
           'BarycenterMapping', 'Berendsen', 'BondStretchAnalyses',
           'BoxHeatUp', 'BoxOscillatingRampModifier', 'BoxScale',
           'CalibrationMetrics', 'Check', 'CollectMDSteps',
           'ComparePredictions', 'ConfigurationSelection', 'ConnectivityCheck',
           'DebugCheck', 'DensityCheck', 'DipoleHistogram', 'EnergyHistogram',
           'EnergySpikeCheck', 'EnergyUncertaintyHistogram', 'FilterOutlier',
           'FixedBondLengthConstraint', 'FixedLayerConstraint',
           'FixedSphereConstraint', 'Flatten', 'ForceAngles',
           'ForceDecomposition', 'ForceUncertaintyDecomposition',
           'ForcesHistogram', 'ForcesUncertaintyHistogram', 'HasAtoms',
           'HasOrIsAtoms', 'HasSelectedConfigurations', 'HookeanConstraint',
           'IPSNode', 'IndexSelection', 'LangevinThermostat', 'MD22Dataset',
           'MDStability', 'MoveSingleParticle', 'NPTThermostat', 'NaNCheck',
           'NodeWithCalculator', 'NodeWithThermostat', 'Prediction',
           'PredictionMetrics', 'PressureRampModifier', 'ProcessAtoms',
           'ProcessSingleAtom', 'Project', 'RandomSelection', 'RattleAnalysis',
           'RattleAtoms', 'RescaleBoxModifier', 'RotateMolecules',
           'STATIC_PATH', 'SVCRBarostat', 'SplitSelection', 'StressHistogram',
           'SurfaceRasterMetrics', 'SurfaceRasterScan', 'TemperatureCheck',
           'TemperatureOscillatingRampModifier', 'TemperatureRampModifier',
           'ThresholdCheck', 'ThresholdSelection', 'TranslateMolecules',
           'UNION_ATOMS_OR_ATOMS_LST', 'UniformArangeSelection',
           'UniformEnergeticSelection', 'UniformTemporalSelection',
           'VelocityVerletDynamic', 'WrapModifier', 'analysis', 'ase_sim',
           'base', 'bootstrap', 'calc', 'calculators', 'combine',
           'configuration_generation', 'configuration_selection', 'conftest',
           'data_loading', 'datasets', 'docs', 'doctest_namespace', 'dynamics',
           'fields', 'geometry', 'helpers', 'interfaces', 'log', 'md',
           'metrics', 'models', 'nodes', 'project', 'static_data', 'utils',
           'version']
