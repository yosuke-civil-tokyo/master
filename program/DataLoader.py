import csv


class DataLoader:
    def __init__(self):
        self.data = None
        self.variable_names = None

    def load_data_from_csv(self, csv_file_path):
        data = {}
        
        with open(csv_file_path, 'r') as f:
            reader = csv.reader(f)
            headers = next(reader)
            self.variable_names = headers
            
            for h in headers:
                data[h] = []
            
            for row in reader:
                for h, value in zip(headers, row):
                    data[h].append(int(value))
        
        self.data = data

    def get_data(self):
        return self.data, self.variable_names