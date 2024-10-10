class ReceptiveFieldResponse:
    def __init__(self, row, col, timecourse):
        self.row = row
        self.col= col
        self.timecourse = timecourse
        self.best_fit_info = None
