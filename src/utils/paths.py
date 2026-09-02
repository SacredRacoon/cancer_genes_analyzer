from pathlib import Path

class PathManager:
    def __init__(self, config: dict):
        paths = config.get('paths', {})
        self.output_dir = Path(paths.get('output_dir','output'))
        self.models_dir = Path(paths.get('models_dir', self.output_dir / 'models'))
        self.plots_dir = Path(paths.get('plots_dir', self.output_dir / 'plots'))
        self.log_file = Path(paths.get('log_file', self.output_dir / 'logs' / 'app.log'))
        self.raw_data = Path(paths.get('raw_data', 'data/raw/data.csv'))
        self.reports_dir = Path(paths.get('reports_dir', self.output_dir / 'reports'))
    def ensure_dirs(self):
        for dir_path in [self.output_dir, self.models_dir, self.reports_dir, self.plots_dir, self.log_file.parent]:
            dir_path.mkdir(parents=True, exist_ok=True)