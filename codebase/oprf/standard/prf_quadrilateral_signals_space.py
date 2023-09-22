import numpy as np
import cupy as cp
         
# Stimulus Class
from codebase.oprf.standard.prf_stimulus import Stimulus

# Receptive Field Response Class
from codebase.oprf.standard.prf_receptive_field_response import ReceptiveFieldResponse

class QuadrilateralSignalsSpace:
    def __init__(self
                 , grid_nRows
                 , grid_nCols
                 , sigma
                 , stimulus : Stimulus
                 ):
        self.grid_nRows = grid_nRows
        self.grid_nCols = grid_nCols
        self.sigma = sigma        
        self.stimulus = stimulus
        self.gaussian_meshgrid_X = None
        self.gaussian_meshgrid_Y = None
        self.custom_space_meshgrid_X  = None
        self.custom_space_meshgrid_Y = None
        self.custom_space_col_points  = None
        self.custom_space_row_points = None
        self.data = np.zeros((grid_nRows, grid_nCols), dtype=ReceptiveFieldResponse)        
            
    def _generate_meshgrids(self):
        # For Gaussian Meshgrid: This meshgrid must have the same size as the stimulus. The 2D Gaussian curves generated using these X and Y meshgrids are used to model a pRF response
        gaussian_curve_x_values = np.linspace( - self.stimulus.size_in_degrees, + self.stimulus.size_in_degrees, self.stimulus.resampled_hrf_convolved_data.shape[0])
        gaussian_curve_y_values = np.linspace( - self.stimulus.size_in_degrees, + self.stimulus.size_in_degrees, self.stimulus.resampled_hrf_convolved_data.shape[1])
        self.gaussian_meshgrid_X, self.gaussian_meshgrid_Y  = np.meshgrid(gaussian_curve_x_values, gaussian_curve_y_values)

        # For Test Locations Grid Meshgrid: These X and Y meshgrids below contains the MEAN POSITIONS for which we would like to generate the "modelled responses". 
        # The size/shape of these meshgrids depends on the CUSTOM SPACE (for test, search or whatever) dimensions
        self.custom_space_col_points = np.linspace(-self.stimulus.size_in_degrees, self.stimulus.size_in_degrees, self.grid_nCols) 
        self.custom_space_row_points = np.linspace(-self.stimulus.size_in_degrees, self.stimulus.size_in_degrees, self.grid_nRows) 
        self.custom_space_meshgrid_X, self.custom_space_meshgrid_Y = np.meshgrid(self.custom_space_col_points, self.custom_space_row_points)

    def _generate_2d_gaussian(self, mean_x, mean_y):                            
        Z = np.exp(-(self.gaussian_meshgrid_X - mean_x)**2 / (2 * self.sigma**2)) * np.exp(-(self.gaussian_meshgrid_Y - mean_y)**2 / (2 * self.sigma**2))
        return Z
    
    # Generate the timecourse for a Gaussian Curve (for a particular pixel location)
    # by taking the Stimulus frames into account
    def _generate_pixel_location_time_course(self, Z):
        time_points = self.stimulus.resampled_hrf_convolved_data.shape[2]
        area_under_gaussian = np.zeros(time_points)

        for t in range(time_points):            
            # Apply mask to the Gaussian curve
            masked_gaussian = Z * self.stimulus.resampled_hrf_convolved_data[:, :, t]

            # Compute area under the Gaussian curve after applying the mask
            area_under_gaussian[t] = np.sum(masked_gaussian)
            
        return area_under_gaussian

    def generate_model_responses(self):
        self._generate_meshgrids()
                
        # Create Expected Responses Grid: Traverse through all locations of the defined grid
        for row in range(self.grid_nRows):
            for col in range(self.grid_nCols):
                pRF = self._generate_2d_gaussian(mean_x= self.custom_space_meshgrid_X[row][col], mean_y= self.custom_space_meshgrid_Y[row][col])
                pixel_location_timecourse = self._generate_pixel_location_time_course(pRF)
                pixel_location_timecourse /= pixel_location_timecourse.max()                
                expected_receptive_field_reponse = ReceptiveFieldResponse(row=row, col=col, timecourse=pixel_location_timecourse)  
                self.data[row][col] = expected_receptive_field_reponse      

        print("computed model signals....")    

    def generate_model_responses_Sid_first_version_superslow(self):
        self._generate_meshgrids()

        # apply masking
        # ...computes the sum along axis 2 (i.e. along the time-direction). If stimulus never reaches a particular location (row, col), the sum for that location across all time-points would be zero 
        mask = self.stimulus.resampled_data.sum(2) > 0
        
        # ...use mask to keep relevant locations in the meshgrids and the hrf-convolved-stimulus
        Xm = self.gaussian_meshgrid_X[mask] # the shape (e.g. (7877,)) of the resulting array Xm will depend on how many True values there are in the mask array
        Ym = self.gaussian_meshgrid_Y[mask] # shapee e.g. (7877,)
        masked_hrf_convolved_stimulus = self.stimulus.resampled_hrf_convolved_data[mask] # shape e.g. (7877, 300)
                
        # Create Expected Responses Grid: Traverse through all locations of the defined grid
        for row in range(self.grid_nRows):
            for col in range(self.grid_nCols):
                mu_x = self.custom_space_meshgrid_X[row][col]
                mu_y = self.custom_space_meshgrid_Y[row][col]
                sigma = self.sigma
                pRF = np.exp(- ((Xm-mu_x)**2 + (Ym-mu_y)**2) / (2 * sigma**2)) #2d Gaussian
                pixel_location_timecourse = pRF @ masked_hrf_convolved_stimulus 
                pixel_location_timecourse /= pixel_location_timecourse.max()                
                expected_receptive_field_reponse = ReceptiveFieldResponse(row=row, col=col, timecourse=pixel_location_timecourse)  
                self.data[row][col] = expected_receptive_field_reponse      

        print("computed model signals....")       

    def generate_model_responses_DAVID_fast_but_partial(self):
        self._generate_meshgrids()

        # apply masking
        # ...computes the sum along axis 2 (i.e. along the time-direction). If stimulus never reaches a particular location (row, col), the sum for that location across all time-points would be zero 
        mask = self.stimulus.resampled_data.sum(2) > 0
        
        # ...use mask to keep relevant locations in the meshgrids and the hrf-convolved-stimulus
        Xm = self.gaussian_meshgrid_X[mask] # the shape (e.g. (7877,)) of the resulting array Xm will depend on how many True values there are in the mask array
        Ym = self.gaussian_meshgrid_Y[mask] # shapee e.g. (7877,)
        masked_hrf_convolved_stimulus = self.stimulus.resampled_hrf_convolved_data[mask] # shape e.g. (7877, 300)
                
        # Create Expected Responses Grid: Traverse through all locations of the defined grid
        # for row in range(self.grid_nRows):
        #     for col in range(self.grid_nCols):
        #         mu_x = self.custom_space_meshgrid_X[row][col]
        #         mu_y = self.custom_space_meshgrid_Y[row][col]
        sigma = self.sigma

        pRF = np.exp(- ((Xm[:,None] - self.custom_space_meshgrid_X.flatten()[None,:])**2 + (Ym[:,None] - self.custom_space_meshgrid_Y.flatten()[None,:])**2) / (2 * sigma**2)) #2d Gaussian
        pixel_location_timecourse = pRF.T @ masked_hrf_convolved_stimulus 
        pixel_location_timecourse /= pixel_location_timecourse.max(1)[:,None]

        # NOTE: This needs to be corrected, but now I don't have "row", "col" info, therefore, need to use flatten indices
        # expected_receptive_field_reponse = ReceptiveFieldResponse(row=row, col=col, timecourse=pixel_location_timecourse)  
        # self.data[row][col] = expected_receptive_field_reponse      

        print("computed model signals....")    
        
    def gpu_cupy_generate_model_responses(self):
        self._generate_meshgrids()

        # apply masking
        mask = cp.sum(self.stimulus.resampled_data, axis=2) > 0

        # Use mask to keep relevant locations in the meshgrids and the hrf-convolved-stimulus
        Xm = self.gaussian_meshgrid_X[mask]
        Ym = self.gaussian_meshgrid_Y[mask]
        masked_hrf_convolved_stimulus = self.stimulus.resampled_hrf_convolved_data[mask]

        sigma = self.sigma

        # Calculate the 2D Gaussian
        Xm_cp = cp.asarray(Xm[:, None])
        Ym_cp = cp.asarray(Ym[:, None])
        custom_space_meshgrid_X_flat_cp = cp.asarray(self.custom_space_meshgrid_X.flatten()[None, :])
        custom_space_meshgrid_Y_flat_cp = cp.asarray(self.custom_space_meshgrid_Y.flatten()[None, :])

        pRF = cp.exp(-((Xm_cp - custom_space_meshgrid_X_flat_cp) ** 2 + (Ym_cp - custom_space_meshgrid_Y_flat_cp) ** 2) / (2 * sigma**2))

        # Transpose the pRF for matrix multiplication
        pRF_T = cp.transpose(pRF)

        # Matrix multiplication
        #pixel_location_timecourse = pRF_T @ masked_hrf_convolved_stimulus
        pixel_location_timecourse = pRF_T @ cp.asarray(masked_hrf_convolved_stimulus)

        # Normalize by dividing by the maximum along axis 1
        max_values = cp.max(pixel_location_timecourse, axis=1)[:, None]
        pixel_location_timecourse /= max_values

        # debug
        # plt.plot(pixel_location_timecourse[0].get())

        print("computed model signals....")