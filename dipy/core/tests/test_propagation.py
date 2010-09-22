import os
import numpy as np
import dipy as dp
from dipy.core.track_propagation import FACT_DeltaX
from dipy.core.reconstruction_performance import ndarray_offset
import nibabel as ni
from os.path import join as opj
from time import time


from nose.tools import assert_true, assert_false, \
     assert_equal, assert_raises

from numpy.testing import assert_array_equal, assert_array_almost_equal


def test_fact():

    #read bvals,gradients and data
    bvals=np.load(opj(os.path.dirname(__file__), \
                          'data','small_64D.bvals.npy'))
    gradients=np.load(opj(os.path.dirname(__file__), \
                              'data','small_64D.gradients.npy'))    
    img =ni.load(os.path.join(os.path.dirname(__file__),\
                                  'data','small_64D.nii'))
    data=img.get_data()    

    print(data.shape)    
    gqs = dp.GeneralizedQSampling(data,bvals,gradients)       
    #ten = dp.Tensor(data,bvals,gradients,thresh=50)
    
    t1=time()
    FD=dp.FACT_Delta(gqs.QA,gqs.IN,seeds_no=5000)
    T1=FD.tracks
    t2=time()    
    print 'I', t2-t1, 'time.'    

    #return

    T2=FACT_DeltaX(gqs.QA,gqs.IN,seed_list=FD.seed_list).tracks
    t3=time()    
    print 'X', t3-t2, 'time.'

    print (t3-t2)/(t2-t1), ' ratio '   
    
    print(len(T1))
    print(len(T2))

    from dipy.core.track_metrics import length

    print('length T1',sum([length(t) for t in T1]))
    print('length T2',sum([length(t) for t in T2]))

    print(gqs.QA[1,4,8,0])
    print(gqs.QA.ravel()[ndarray_offset(np.array([1,4,8,0]),np.array(gqs.QA.strides),4,8)])

    #'''
    for i in range(len(FD.seed_list)):

        print('seed',FD.seed_list[i])

        print('T1[%d]' % i)
        print T1[i]

        print('T2[%d]' % i)
        print T2[i]
    #'''

    from dipy.viz import fos
    #import numpy as np

    r=fos.ren()
    fos.add(r,fos.line(T1,fos.red,opacity=0.1))

    for se in FD.seed_list:
        x,y,z=se
        #fos.add(r,fos.sphere((x,y,z),color=(1,1,0)))
    
    T2=[t+np.array([10,0,0]) for t in T2]
    fos.add(r,fos.line(T2,fos.green,opacity=0.1))

    for se in FD.seed_list:
        x,y,z=se+np.array([10,0,0])
        #fos.add(r,fos.sphere((x,y,z),color=(1,1,0)))
    

    
    fos.show(r)
    