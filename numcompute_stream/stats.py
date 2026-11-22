"""
Provides descriptive statistics for the numerical arrays for analysis 
components.

It calculates basic statistics such as mean, median, standard deviation,
min and max. It also groups observations into bins, counts how many 
observations fall into each interval and also uses quantiles to identify
values that divide ordered observations into specified proportions. 

The basic statistics must follow axis-wise statistics with clear dimension
and shape behaviour, and the quantiles must also handle NaN values 
appropriately.  
"""
import numpy as np

class StreamStats:

    def __init__(self):

        self.mean = 0.0  # running mean
        self.var = 0.0  # running variance
        self.M2 = 0.0   # running sum of squared deviations
        self.count = 0 # cumulative items processed from previous chunks

        # for calculation of streamed min and max
        self.current_max = -np.inf
        self.current_min = np.inf

    def max_stream(self, chunk_x, axis=0):
        self.current_max = -np.inf
        """
            Calculates the max for the inputed chunk data, updates with new batches

            Parameters:
                data: a numpy array containing the data to calculate the min of
                axis: the axis to operate on, defaults to 0


            Returns:
                A scalar or an array containing the resulting mins

            Raises:
                

            Time Complexity:
                O(n) where n is the number of elements of the data

            Space Complexity:
                O(n) where n is the number of elements of the data
        """
        chunk_running_max = np.maximum.accumulate(chunk_x, axis=axis)

        updated_max = np.maximum(chunk_running_max, self.current_max)

        self.current_max = np.take(updated_max, -1, axis=axis)
        return updated_max

    def min_stream(self, chunk_x, axis=0):
        self.current_min = np.inf
        """
            Calculates the min for the inputed chunk data, updates with new batches

            Parameters:
                data: a numpy array containing the data to calculate the min of
                axis: the axis to operate on, defaults to 0


            Returns:
                A scalar or an array containing the resulting mins

            Raises:

            Time Complexity:
                O(n) where n is the number of elements of the data

            Space Complexity:
                O(n) where n is the number of elements of the data
        """
        chunk_running_min = np.minimum.accumulate(chunk_x)

        updated_min = np.minimum(chunk_running_min, self.current)

        self.current_min = np.take(updated_min, -1, axis=axis)
        return updated_min

    def update_chunks(self, chunk_x):
        """
           Takes in a 1D NumpyArray and updates mean and variance using Welford's 
           Algorithm and Chan et al's Parallel Algorithm 
           https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm

           Parameters:
              self: Existing chunk/s that have already been processed
              chunk_x: the next chunk to be processed

            Returns:
                Running mean and variance

            Time Complexity:
                O(n) where n is the elements being processed

            Space Complexity:
                O(1)

        """
        if chunk_x.size == 0:
            return
        else:
            n_b = chunk_x.shape[0]  # items in the next chunk to be processed, chunk_x

        mean_b = np.mean(chunk_x, axis=0)

        total_n = self.count + n_b  

        delta = mean_b - self.mean 

        M2_b = np.sum((chunk_x - mean_b) ** 2, axis=0)

        if self.count == 0:
            self.count = n_b
            self.mean = mean_b
            self.M2 = M2_b
            self.var = self.M2 / self.count
            return self
            
        # mean and variance
        self.mean = self.mean + delta * n_b/total_n
        self.M2 = self.M2 + M2_b + (delta ** 2) * (self.count * n_b / total_n)
        self.var = self.M2 / total_n

        self.count = total_n
    
        return self 

    def chunk_stan_dev(self) -> float:
        print("DEBUG self.var:", repr(self.var), type(self.var))
        stan_dev = np.sqrt(self.var)
        return np.where(stan_dev == 0.0, 1.0, stan_dev)
    
class StreamQuantileP2:
    """ Implements streaming quantiles by using the P2 Algorithm
    https://dl.acm.org/doi/epdf/10.1145/4372.4378 

    Jain, R., & Chlamtac, I. (1985). The P^2 algorithm for dynamic calculation of quantiles and histograms without 
    storing observations. Communications of the ACM, 28(10), 1076–1085. https://doi.org/10.1145/4372.4378
    """
    
    def __init__(self, p):
        self.p = p # percentile
        self.n = np.arange(1,6)  # marker positions
        self.n_positions = np.array([1, 1 + 2*p, 1 + 4*p, 3 + 2*p, 5], dtype=np.float64) # marker desired positions (ns)
        self.dns = np.array([0, p/2, p, (1 + p)/2, 1], dtype=np.float64) # precalculated increments of desired positions added to current values after each  observation
        self.q_heights = np.zeros(5, dtype=np.float64) # q/marker heights 
        
        self.count = 0 # cumulative items processed from previous chunks
        self.obs = [] # first 5 observations
    
    def update_chunks(self, chunk_x):
        """ Runs the P2 algorithm and saves the result"""
        for x in chunk_x:
            self.add(x)
        return self

    
    def add(self, x): #
        """
        Using P2 Algorithm on a single observation

        Parameters:
            x: The individual observation in chunk x


        Raises:
            ValueError: if no data is provided
            TypeError: if provided data has strings, or none
            Exception: If Step 1 of the algorithm fails to produce valid result

        Time Complexity:
            O(n) where n is the number of elements of the data

        Space Complexity:
            O(1) Has constant space



        """
        # initializing values
        p = self.p
        n = self.n
        q_heights = self.q_heights
        n_positions = self.n_positions
        dns = self.dns
        count = self.count
        
        if len(self.q_heights) <5:
            self.q_heights.append(x)
            self.q_heights = sorted(self.q_heights)
            return self

        # Step 1 of the Algorithm
        if x<= self.q_heights[0]:
            self.q_heights[0] = x # new minimum
            k = 0
        elif x >= self.q_heights[4]:
            self.q_heights[4] = x
            k = 3
        elif x < self.q_heights[1]:
            k = 0
        elif x < self.q_heights[2]:
            k = 1
        elif x < self.q_heights[3]:
            k = 2 
        elif x < self.q_heights[4]:
            k = 3
        else:
            raise Exception("Step 1 of the algorithm has produced an invalid result, cannot compute k")

        # Step 2 of the Algorithm, increment position of markers k+1 through 5
        # Update desired positions for all markers
        for i in range(k+1, 5):
            self.n[i] += 1
        for i in range (5):
            self.n_positions[i] += self.dns[i]

        # Step 3 between minimum and max, adjust heights for markers 2-4
        for i in range(1,4):
            d = self.n_positions[i] - self.n[i]
            if (d >= 1 and self.n[i + 1] - n[i] > 1) or (d <= -1 and self.n[i - 1] - self.n[i] < -1):
                # select d to be the sign
                d_sign = 1 if d > 0 else -1
                # Condition to check whether we use parabolic or linear  equation
                q_para = self._parabolic(i, d_sign)
                if (self.q_heights[i-1] < q_para) and (q_para < self.q_heights[i+1]):
                    self.q_heights[i] = q_para
                else:
                    self.q_heights[i] = self._linear(i, d_sign)
                self.n[i] +=d_sign
        
    def score(self):
        """Estimates the percentile based on pre-calculated steps

            Returns:
                q_heights: index closest to estimated percentile
        """

        if self.p == 0:
            return self.q_heights[0]
        elif self.p == 1:
            return self.q_heights[4]

        # selecting an index that's closest to the desired percentile
        elif len(self.q_heights ) < 5:
            index = int(len(self.q_heights) * self.p) 
            return self.q_heights[index]

        return self.q_heights[2]    
    
    def _parabolic(self, i:int, d: np.double) -> float:
            """
                Quantile if a parabolic equation is needed in step 3

                Returns: Parabolic estimate
            """
            q, n = self.q_heights, self.n
            
            return q[i] + d / (n[i + 1] - n[i - 1]) * (
            (n[i] - n[i - 1] + d) * (q[i + 1] - q[i]) / (n[i + 1] - n[i])
            + (n[i + 1] - n[i] - d) * (q[i] - q[i - 1]) / (n[i] - n[i - 1])
        )
    
    def _linear(self, i:int, d: int) -> float:
        """
            Quantile if a linear  equation is needed in step 3

            Returns: linear estimate

        """
        q, n = self.q_heights, self.n
        return q[i] + d * (q[i + d] - q[i]) / (n[i + d] - n[i])


def mean(data, axis=0):
    """
        Calculates the statistical mean for the inputed data

        Parameters:
            data: a numpy array containing the data to calculate the mean of
            axis: the axis to operate on, defaults to 0

        Returns:
            A scalar or an array containing the resulting means

        Raises:
            ValueError: if no data is provided
            AxisError: if provided axis index is out of bounds
            TypeError: if provided data has strings, or none
        
        Time Complexity:
            O(n) where n is the number of elements of the data

        Space Complexity:
            O(n) where n is the number of elements of the data
    """
    return np.nanmean(data,axis)

def median(data,axis=0):
    """
        Calculates the stastical median for the inputed data

        Parameters:
            data: a numpy array containing the data to calculate the median of
            axis: the axis to operate on, defaults to 0
            

        Returns:
            A scalar or an array containing the resulting median

        Raises:
            ValueError: if no data is provided
            AxisError: if provided axis index is out of bounds
            TypeError: if provided data has strings, or none

        Time Complexity:
            O(n) where n is the number of elements of the data

        Space Complexity:
            O(n) where n is the number of elements of the data
    """  
    return np.nanmedian(data,axis)

def stan_dev(data,axis=0):
    """
        Calculates the stastical standard deviation for the inputed data

        Parameters:
            data: a numpy array containing the data to calculate the SD of
            axis: the axis to operate on, defaults to 0

        Returns:
            A scalar or an array containing the resulting SD

        Raises:
            ValueError: if no data is provided
            AxisError: if provided axis index is out of bounds
            TypeError: if provided data has strings, or none
        
        Time Complexity:
            O(n) where n is the number of elements of the data

        Space Complexity:
            O(n) where n is the number of elements of the data
    """ 
    return np.nanstd(data,axis)

class StreamHistogram:
    """
        Attributes:
            start: start of the bin range
            end: end of the bin range
            bins: number of bins
            edges: edges of the bins
            counts: counts of values in each bin.
        
        Time Complexity:
            O(m log n ) where n is the number of elements of the data and m is the number of bins

        Space Complexity:
            O(n) where n is the amount of bins
    """

    def __init__ (self, start, end , bins=10):
        self.start = start
        self.end = end
        self.bins = bins
        self.edges = np.histogram_bin_edges([], bins=bins, range=(start, end))
        self.counts = np.zeros(bins)

    def histogram(self, chunk_x):
        """
        Updates the histogram with a new value

         Parameters:
            chunk_x: Chunked values on the histogram

        Returns:
            Counts for each edge of the histogram.

        Raises:
            ValueError: if chunk is empty

        Time Complexity:
            O(m log n ) where n is the number of elements of the data and m is the number of bins

        Space Complexity:
            O(n) where n is the amount of bins
            
        """
        if chunk_x is None:
            raise ValueError("Chunk cannot be empty")

        chunk_x = chunk_x[~np.isnan(chunk_x)]
        self.counts+= np.histogram(chunk_x, bins=self.edges)[0]
        return self.counts, self.edges





def quantile(data, q):
    """
        Groups the data into the probability or sequence of probabilities to compute

        Parameters:
            data: a numpy array containing the data to calculate the mean of
            q: the probability or sequence of probabilites the data will be grouped into
            

        Returns:
            Scalar or array of the quantiles

        Raises:
            ValueError: if data is empty
        
        Time Complexity:
            O(bn) where n is the number of elements of the data and b is the number of bins

        Space Complexity:
            O(n) where n is the number of elements of the data
    """
    return np.nanquantile(data, q)