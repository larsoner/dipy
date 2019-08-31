import numpy as np
import dipy.reconst.shm as shm
from dipy.denoise.localpca import mppca
from dipy.data import (fetch_cfin_multib, read_cfin_dwi)
from dipy.core.gradients import gradient_table
from dipy.segment.mask import median_otsu
import dipy.direction.peaks as dp
from dipy.reconst.mcsd import MultiShellResponse
from dipy.reconst.csdeconv import auto_response
from dipy.segment.tissue import TissueClassifierHMRF
from dipy.sims.voxel import (single_tensor)
from dipy.data import default_sphere
from dipy.core.gradients import GradientTable
import dipy.reconst.dti as dti
from dipy.reconst.mcsd import MultiShellDeconvModel
from dipy.viz import window, actor
from dipy.data import get_sphere
sphere = get_sphere('symmetric724')


fetch_cfin_multib()
img, gtab = read_cfin_dwi()
data = img.get_data()
affine = img.affine

bvals = gtab.bvals
bvecs = gtab.bvecs
sel_b = np.logical_or(np.logical_or(bvals == 0, bvals == 1000), bvals == 2000)
data = data[..., sel_b]
gtab = gradient_table(bvals[sel_b], bvecs[sel_b])
b0_mask, mask = median_otsu(data, median_radius=2, numpass=1, vol_idx=[0, 1])

denoised_arr = mppca(data, mask=mask, patch_radius=2)

model = shm.QballModel(gtab, 8)

peaks = dp.peaks_from_model(model=model, data=data,
                            relative_peak_threshold=.5,
                            min_separation_angle=25,
                            sphere=sphere, mask=mask)

ap = shm.anisotropic_power(peaks.shm_coeff)

tenmodel = dti.TensorModel(gtab)
tenfit = tenmodel.fit(data)

nclass = 3
beta = 0.1

FA = tenfit.fa
MD = tenfit.md

hmrf = TissueClassifierHMRF()
initial_segmentation, final_segmentation, PVE = hmrf.classify(ap, nclass,
                                                              beta)


csf = PVE[..., 0]
cgm = PVE[..., 1]


indices_csf = np.where(((FA < 0.2) & (csf > 0.95)))
indices_cgm = np.where(((FA < 0.2) & (cgm > 0.95)))

selected_csf = np.zeros(FA.shape, dtype='bool')
selected_cgm = np.zeros(FA.shape, dtype='bool')

selected_csf[indices_csf] = True
selected_cgm[indices_cgm] = True

csf_md = np.mean(tenfit.md[selected_csf])
cgm_md = np.mean(tenfit.md[selected_cgm])


response, ratio = auto_response(gtab, data, roi_radius=10, fa_thr=0.7)
evals_d = response[0]


# TODO: will remove once Jon's PR is merged
def sim_response(sh_order=8, bvals=bvals, evals=evals_d, csf_md=csf_md,
                 gm_md=cgm_md):
    bvals = np.array(bvals, copy=True)
    evecs = np.zeros((3, 3))
    z = np.array([0, 0, 1.])
    evecs[:, 0] = z
    evecs[:2, 1:] = np.eye(2)

    n = np.arange(0, sh_order + 1, 2)
    m = np.zeros_like(n)

    big_sphere = default_sphere.subdivide()
    theta, phi = big_sphere.theta, big_sphere.phi

    B = shm.real_sph_harm(m, n, theta[:, None], phi[:, None])
    A = shm.real_sph_harm(0, 0, 0, 0)

    response = np.empty([len(bvals), len(n) + 2])
    for i, bvalue in enumerate(bvals):
        gtab = GradientTable(big_sphere.vertices * bvalue)
        wm_response = single_tensor(gtab, 1., evals, evecs, snr=None)
        response[i, 2:] = np.linalg.lstsq(B, wm_response)[0]

        response[i, 0] = np.exp(-bvalue * csf_md) / A
        response[i, 1] = np.exp(-bvalue * gm_md) / A

    return MultiShellResponse(response, sh_order, bvals)


response_mcsd = sim_response(sh_order=8, bvals=bvals, evals=evals_d,
                             csf_md=csf_md, gm_md=cgm_md)

mcsd_model = MultiShellDeconvModel(gtab, response_mcsd)

mcsd_fit = mcsd_model.fit(data[:, :, 10:10+1])
mcsd_odf = mcsd_fit.odf(sphere)
fodf_spheres = actor.odf_slicer(mcsd_odf, sphere=sphere, scale=0.01,
                                norm=False, colormap='coolwarm')
interactive = True
ren = window.Renderer()
ren.add(fodf_spheres)

if interactive:
    window.show(ren)
