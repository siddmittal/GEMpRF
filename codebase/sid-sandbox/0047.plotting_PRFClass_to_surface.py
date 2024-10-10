import sys
# sys.path.append('/ceph/mri.meduniwien.ac.at/departments/physics/fmrilab/home/dlinhardt/pythonclass')
sys.path.append("Z:\\home\\dlinhardt\\pythonclass")


from PRFclass import PRF


subjects = ['001', '002']
sessions = ['002', '003']
basePath = "Y:\\data\\"

vista = PRF.from_docker('stimsim23', 'sidtest', '001', 'bar', '02', analysis='03', baseP=basePath, orientation='MP', method='vista')
vista.maskROI('V1')
vista.maskVarExp(0.1)
vista.plot_toSurface()