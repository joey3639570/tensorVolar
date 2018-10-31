class Converge:
    def __init__(self, sample_len=15, epsilon=0.1, ref_index='middle', epsilon_relative=1):
        self.SAMPLING_LEN = sample_len
        self.EPSILON = epsilon
        self.EPSILON_RELATIVE = 0.001
        if ref_index == 'head':
            self.REFERENCE_INDEX = 0
        elif ref_index == 'middle':
            self.REFERENCE_INDEX = self.SAMPLING_LEN//2
        elif ref_index == 'end':
            self.REFERENCE_INDEX = self.SAMPLING_LEN

class G14:
    def __init__(self):
        '''Parameters to obtain datasets'''
        # Dataset name class 'Datasets' in 'readDataset' 
        self.DATASET_NAME = 'G14'
        self.CONVERGE_MIN_TEMPERATURE = 250
        # 
        self.LAST_DATA_DECIDE_INDEX = 50
        '''Converge Definition'''
        self.MOVING_LEN = 1
        self.COOL_LEN = 20
        self.CONVERGE = Converge(sample_len=10, epsilon=0.1, ref_index='middle')
        self.MODEL_AVR_OFFSET = -1.24

        '''Parameters for planB'''
        # Align other data by this index(time)
        self.ALIGN_INDEX = 50
        # Use this index (sample) to align other sample
        self.BASE_SAMPLE_INDEX = 0
        # Use this index(sensor) to align other sample (count from back)
        self.SENSOR_INDEX = 0
        # Search for align index in this index range
        self.SEARCH_RANGE = self.ALIGN_INDEX + 50
        self.GOOD_SAMPLES = [False, True, False, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, True, True, True, True, True, True, True, True, False, True, True, True, True, True, True, True, True, True, True]
class G35:
    def __init__(self):
        '''Parameters to obtain datasets'''
        # Dataset name class 'Datasets' in 'readDataset' 
        self.DATASET_NAME = 'G35'
        self.CONVERGE_MIN_TEMPERATURE = 350
        # 
        self.LAST_DATA_DECIDE_INDEX = 30
        '''Converge Definition'''
        self.MOVING_LEN = 10
        self.COOL_LEN = 20
        self.CONVERGE = Converge(sample_len=10, epsilon=0.3, ref_index='middle')

        '''Parameters for planB'''
        # Align other data by this index(time)
        self.ALIGN_INDEX = 100
        # Use this index (sample) to align other sample
        self.BASE_SAMPLE_INDEX = 0
        # Use this index(sensor) to align other sample (count from back)
        self.SENSOR_INDEX = 0
        # Search for align index in this index range
        self.SEARCH_RANGE = 200


class G57:
    def __init__(self):
        '''Parameters to obtain datasets'''
        # Dataset name class 'Datasets' in 'readDataset' 
        self.DATASET_NAME = 'G57'
        self.CONVERGE_MIN_TEMPERATURE = 340
        # 
        self.LAST_DATA_DECIDE_INDEX = 50
        '''Converge Definition'''
        self.MOVING_LEN = 1
        self.COOL_LEN = 20
        self.CONVERGE = Converge(sample_len=10, epsilon=0.1, ref_index='middle')
        self.MODEL_AVR_OFFSET = 20.5
        '''Parameters for planB'''
        # Align other data by this index(time)
        self.ALIGN_INDEX = 100
        # Use this index (sample) to align other sample
        self.BASE_SAMPLE_INDEX = 0
        # Use this index(sensor) to align other sample (count from back)
        self.SENSOR_INDEX = 0
        # Search for align index in this index range
        self.SEARCH_RANGE = 150

        self.GOOD_SAMPLES = [True, True, False, True, True, True, True, True, True, True, True, True, True, True, False, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, True, True, True, True, True, True, False, True, True, True, True, False, True, True, True, False]
class G30:
    def __init__(self):
        '''Parameters to obtain datasets'''
        # Dataset name class 'Datasets' in 'readDataset' 
        self.DATASET_NAME = 'G30'
        self.CONVERGE_MIN_TEMPERATURE = 250
        # 
        self.LAST_DATA_DECIDE_INDEX = 30
        '''Converge Definition'''
        self.MOVING_LEN = 10
        self.COOL_LEN = 20
        self.CONVERGE = Converge(sample_len=10, epsilon=0.1, ref_index='middle')
        self.MODEL_AVR_OFFSET = -1.24

        '''Parameters for planB'''
        # Align other data by this index(time)
        self.ALIGN_INDEX = 100
        # Use this index (sample) to align other sample
        self.BASE_SAMPLE_INDEX = 0
        # Use this index(sensor) to align other sample (count from back)
        self.SENSOR_INDEX = 0
        # Search for align index in this index range
        self.SEARCH_RANGE = 200

class G29:
    def __init__(self):
        '''Parameters to obtain datasets'''
        # Dataset name class 'Datasets' in 'readDataset' 
        self.DATASET_NAME = 'G29'
        self.CONVERGE_MIN_TEMPERATURE = 350
        # 
        self.LAST_DATA_DECIDE_INDEX = 30
        '''Converge Definition'''
        self.MOVING_LEN = 10
        self.COOL_LEN = 20
        self.CONVERGE = Converge(sample_len=10, epsilon=0.1, ref_index='middle')
        self.MODEL_AVR_OFFSET = -1.24

        '''Parameters for planB'''
        # Align other data by this index(time)
        self.ALIGN_INDEX = 100
        # Use this index (sample) to align other sample
        self.BASE_SAMPLE_INDEX = 0
        # Use this index(sensor) to align other sample (count from back)
        self.SENSOR_INDEX = 0
        # Search for align index in this index range
        self.SEARCH_RANGE = 200

class G44:
    def __init__(self):
        '''Parameters to obtain datasets'''
        # Dataset name class 'Datasets' in 'readDataset' 
        self.DATASET_NAME = 'G44'
        self.CONVERGE_MIN_TEMPERATURE = 350
        # 
        self.LAST_DATA_DECIDE_INDEX = 30
        '''Converge Definition'''
        self.MOVING_LEN = 10
        self.COOL_LEN = 20
        self.CONVERGE = Converge(sample_len=10, epsilon=0.1, ref_index='middle')

        '''Parameters for planB'''
        # Align other data by this index(time)
        self.ALIGN_INDEX = 100
        # Use this index (sample) to align other sample
        self.BASE_SAMPLE_INDEX = 0
        # Use this index(sensor) to align other sample (count from back)
        self.SENSOR_INDEX = 0
        # Search for align index in this index range
        self.SEARCH_RANGE = 200

class G44:
    def __init__(self):
        '''Parameters to obtain datasets'''
        # Dataset name class 'Datasets' in 'readDataset' 
        self.DATASET_NAME = 'G44'
        self.CONVERGE_MIN_TEMPERATURE = 350
        # 
        self.LAST_DATA_DECIDE_INDEX = 30
        '''Converge Definition'''
        self.MOVING_LEN = 10
        self.COOL_LEN = 20
        self.CONVERGE = Converge(sample_len=10, epsilon=0.1, ref_index='middle')

        '''Parameters for planB'''
        # Align other data by this index(time)
        self.ALIGN_INDEX = 100
        # Use this index (sample) to align other sample
        self.BASE_SAMPLE_INDEX = 0
        # Use this index(sensor) to align other sample (count from back)
        self.SENSOR_INDEX = 0
        # Search for align index in this index range
        self.SEARCH_RANGE = 200


