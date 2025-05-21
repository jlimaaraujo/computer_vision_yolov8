import pandas as pd

class Metrics:
    def __init__(self, fps):
        self.people_data = {}  # {id: {'entrada': frame, 'último': frame}}
        self.fps = fps

    def update(self, track_id, frame_count):
        if track_id not in self.people_data:
            self.people_data[track_id] = {'entrada': frame_count, 'último': frame_count}
        else:
            self.people_data[track_id]['último'] = frame_count

    def export_csv(self, path):
        data = []
        for id, info in self.people_data.items():
            tempo = (info['último'] - info['entrada']) / self.fps
            data.append({'ID': id, 'Tempo (s)': round(tempo, 2)})

        df = pd.DataFrame(data)
        df.to_csv(path, index=False)

    def get_avg_time(self):
        tempos = [(info['último'] - info['entrada']) / self.fps for info in self.people_data.values()]
        return sum(tempos) / len(tempos) if tempos else 0
