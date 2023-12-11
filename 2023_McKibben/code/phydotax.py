#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

proportions = phydotax(known_subset, unknown_subset)

Runs PHYDOTax function. ** Note known and unknown dataframes must be normalized and subset with identical approach.**
Specifically the values in the known and unknown indices, as well as the normalization method applied, must be identical. 
Further information and examples in help file below. This is a Python adaptation of the original PHYDOTax code 
by S.L. Palacios in MATLAB.
    
 INPUTS: 
    
    known   =>  The input signature library of remote sensing reflectances (Rrs) representative of anticipated in-water constituents         
                FORMAT REQUIREMENTS --> Pandas DataFrame as follows:
                ** Cells = NORMALIZED Rrs values (float, unitless)
                ** Index, row labels (int or float) = wavelength in number format (e.g. 500, 505, 510... etc), 
                   and this index must match index of the "unknown" dataframe.
                   For example: if phydotax runs are desired from 410nm to 675 nm at 5nm spectral resolution, 
                   the index of both the known and unknown must be subset to [410, 415, 420, 425.........660, 665, 670, 675]
                ** Column labels (str) = the names of constituents represented by the Rrs values (e.g phytoplankton classes) 
 
    unknown =>  The Rrs measurement(s) PHYDOTax is being run on
                FORMAT REQUIREMENTS  --> Pandas DataFrame or Series as follows:
                ** Cells = NORMALIZED Rrs values (float, unitless)
                ** Index, row labels (int or float) = wavelength in number format (e.g. 500, 505, 510... etc), 
                   and this index must match index of the "known" dataframe.
                ** Columns of dataframe are instances of Rrs measurements. There can be 1 or >1 columns (i.e. a Series or DataFrame). 
                ** Column labels (str or int) = unique identifier for each measurement, required if "unknown" has >1 columns. 
                   For example these labels should be station names (str) or number values (int) 
                   that uniquely identify each Rrs measurement PHYDOTax is run on. 
                                     
 RETURNS:
   
    proportions  => dataframe or series of coefficient solutions for each type in the input remote sensing reflectance library  


Version 1.2. Last updated 10/2023 by morgaine.mckibben@nasa.gov

"""
    
def phydotax(known_subset, unknown_subset): 
    
    import numpy as np
    import pandas as pd   
    import gc
    
    # Run the matrix calculations
    # This calculation *is* PHYDOTax. Standard form: X = (A^-1) -dot- B
    #     "The solution to this equation is a best fit approach using least squares minimization and is over determined,
    #     i.e. must have a greater number of wavelengths than taxa" from Palacios 2012  
   
    K = (np.linalg.lstsq(known_subset, known_subset, rcond=None)[0]).T                  
    M = (np.linalg.lstsq(known_subset, unknown_subset, rcond=None)[0]).T        
    
    # Filtering out the negatives
    K1= pd.DataFrame(np.where(K < 1e-3, 0, K), columns = known_subset.columns)
    M_pos = np.where(M < 1e-3, 0, M)                                            #keep as numpy instead of pd with columns for M1 calculation below                               
    
    # Divide PhyDOTax-derived values for "unknown" matrix by the total of each row 
    M1 = M_pos / np.transpose([np.sum(M_pos, axis=1)]*M_pos.shape[1])                              # NP arrays here instead of Pandas
    proportions = pd.DataFrame(M1, columns =known_subset.columns, index=unknown_subset.columns)    # Turning back to pandas dataframe for output
    del K, M, M_pos
    
    return proportions
    
    del proportions, known_subset, unknown_subset
    gc.collect()

    