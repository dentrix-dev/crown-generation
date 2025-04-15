from sampling.PointsCloud.fps_grouping import fbsGrouping
from sampling.PointsCloud.fpsample import fpsample_fps, fps

# Factory to choose the sampling technique use args
SAMPLING_FACTORY = {
    'fpsample':fpsample_fps,
    'fps':fps,
    'mine_fps': fbsGrouping,
}

def get_sampling_technique(technique_name):
    if technique_name in SAMPLING_FACTORY:
        return SAMPLING_FACTORY[technique_name]
    else:
        raise ValueError(f"Sampling technique {technique_name} not recognized")