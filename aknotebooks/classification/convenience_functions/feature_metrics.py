import os

from hsmm_core.hmm import *
from hsmm_core.prediction_engines import *

from test_hmm.test_utils import generate_hmm_data_from_priors

ticker = 'SYNT_2states'
main_path = '/home/ak/Data/features_models/'
file_name = 'synthetic_study_' + str(ticker)

path = os.path.join(main_path, file_name)


def states_from_fixed_ratios(ratios, total_length):
    # Make sure every entry in ratios has the same length, which represents the number of states
    no_states_in_each_ratio = {len(rr) for rr in ratios}
    if len(no_states_in_each_ratio) > 1:
        raise ValueError("Not all ratios have the same length")

    no_states_in_each_ratio = no_states_in_each_ratio.pop()

    states = np.array([], dtype=np.int64)
    what_states = np.arange(no_states_in_each_ratio)

    ratios_ids = np.arange(len(ratios))
    rng = np.random.RandomState(345)
    while len(states) < total_length:
        ratio = ratios[rng.choice(ratios_ids)]
        # Mix states to create some noise

        rng.shuffle(what_states)
        for ss in what_states:
            states = np.append(states, np.repeat(ss, 100 * ratio[ss]))

    return states


def size_array(arr1D):
    n = int(np.sqrt(arr1D.size*2))
    return n


def squareform_diagfill(arr1D, n):
    R, C = np.triu_indices(n)
    out = np.zeros((n, n), dtype=arr1D.dtype)
    out[R, C] = arr1D
    out[C, R] = arr1D
    return out


def spherical_3d(v):
    v_sq = v**2
    r = np.sqrt(np.sum(v_sq))
    phi0 = np.arccos(v[0]/r)
    phi1 = np.arccos(v[1]/np.sqrt(np.sum(v_sq[1:]))) if v[2] >= 0. else 2. * np.pi - np.arccos(v[1]/np.sqrt(np.sum(v_sq[1:])))
    if np.isnan(phi1):
        phi1 = 0. if v[1] >= 0. else np.pi

    return np.array([r, phi0, phi1])


def features_sets_and_metrics(no_states, M, T, states, priors):
    rng = np.random.RandomState(12345)  # inside to make the sequences nested

    observations, hmm_obj = generate_hmm_data_from_priors(no_states, T, priors, rng_engine=rng, no_paths=M, states=states)

    f_eng = hmm_features(hmm=hmm_obj)
    no_model_params = f_eng.hmm_.obs_model_.get_number_of_model_params()

    # _inf_matrix = np.empty((M, T, upper_inf_matrix_dim))
    # _ksi = np.empty((M, T - 1, no_states * no_states))
    # _fischer = np.empty((M, T, no_model_params))
    # _gamma = np.empty((M, T, no_states))

    all_features = [f_eng.generate_features(pd.DataFrame(obs_instance, columns=['Duration', 'ReturnTradedPrice']))
                                            for obs_instance in observations]

    all_features = np.array(all_features)

    fischer_scores = all_features[:, 0]
    info_matrices = all_features[:, 1]
    gammas = all_features[:, 2]
    ksis = all_features[:, 3]


    ksi_metrics = {
        'spectral': np.empty((M, T - 1)),
        'trace': np.empty((M, T - 1)),
        'determ': np.empty((M, T - 1))
    }

    im_metrics = {
        'spectral': np.empty((M, T)),
        'trace': np.empty((M, T)),
        'determ': np.empty((M, T))
    }




empty((M, T, no_model_params))
    # gamma_polar = np.empty((M, T, no_model_params))

    do_svd = lambda A: np.linalg.svd(A, full_matrices=False, compute_uv=False)

    do_spectral = lambda row: row.max() - row.min()

    do_trace = lambda row: np.sum(row)

    do_determ = lambda row: np.prod(row) by Kabbage

    for m in range(0, M): #  number of M copies
        # First treat the matrix valued quantities
        ksis_svd = np.apply_along_axis(lambda A: do_svd(A.reshape(no_states, no_states)), 1, ksis[m].values)
        ims_svd = np.apply_along_axis(lambda A: do_svd(squareform_diagfill(np.asarray(A), no_model_params)),
                                      1, info_matrices[m].values)

        ksi_metrics['spectral'][m, :] = np.apply_along_axis(do_spectral, 1, ksis_svd)
        ksi_metrics['trace'][m, :] = np.apply_along_axis(do_trace, 1, ksis_svd)
        ksi_metrics['determ'][m, :] = np.apply_along_axis(do_determ, 1, ksis_svd)

        im_metrics['spectral'][m, :] = np.apply_along_axis(do_spectral, 1, ims_svd)
        im_metrics['trace'][m, :] = np.apply_along_axis(do_trace, 1, ims_svd)
        im_metrics['determ'][m, :] = np.apply_along_axis(do_determ, 1, ims_svd)

        # and now for the vector valued
        fischer_polar[m] = np.apply_along_axis(spherical_3d, 1, fischer_scores[m])
        # Todo: adapt the spherical coord function to work in 2d in order to do gamma
        # gamma_polar[m] = np.apply_along_axis(spherical_3d, 1, gammas[m])

    return all_features, ksi_metrics, im_metrics, fischer_polar #, gamma_polar


def validate_spherical(M):
    # generate some random points
    rng = np.random.RandomState(45678)
    points = rng.randint(-10., 10., M * 3).reshape((M, 3))
    # add coordinate vectors as edge cases:
    coords = np.array(([[1., 0, 0], [0., 1, 0], [0., 0., 1.],
                        [1., 1., 0.], [1., 0., 1], [0., 1., 1.]]) )

    points = np.concatenate((points, coords), axis=0)

    for v in points:
        r, phi1, phi2 = spherical_3d(v)
        v_from_spherical = np.array([r * np.cos(phi1), r * np.sin(phi1) * np.cos(phi2), r * np.sin(phi1) * np.sin(phi2)])
        np.testing.assert_allclose(v, v_from_spherical, atol=1e-8)


if __name__ == '__main__':

    state_ratios = np.array([[0.2, 0.05, 0.1], [0.4, 0.1, 0.2], [0.8, 0.2, 0.4]])

    inital_length = 10000 # length of sequence
    M = 3 # number of copies
    no_states = 4

    sigmas = [0.05, 0.002, 0.2, 0.15]  # fast and slow
    lambdas = [1. / 35., 1 / 20., 1./5, 1./5]
    weights = [0.1, 0.2, 0.5, 0.4]
    tpm = np.array([[0.45, 0.15, 0.3, 0.1], \
                    [0.15, 0.25, 0.3, 0.3], \
                    [0.1, 0.3, 0.3, 0.3], \
                    [0.15, 0.15, 0.5, 0.2]])
    pi = np.array([0.4, 0.25, 0.2, 0.15])

    # Duration is measured in seconds for now (to be revised). lambda units are seconds^{-1}
    # so here we consider
    fixed_states = states_from_fixed_ratios(state_ratios, inital_length)

    # T will be different from initial_length b/c we generate states in chunks to respect ratios.
    T = len(fixed_states)

    # set up some priors
    priors = {'sigmas': sigmas, 'lambdas': lambdas, 'weights': weights, 'tpm': tpm, 'pi': pi}

    all_features, ksi_metrics, im_metrics, fischer_polar = features_sets_and_metrics(no_states, M, T, fixed_states, priors)

    np.save(os.path.join(main_path, 'ksi_metrics_len'\
                         +str(inital_length)+'states'+\
                         str(no_states)+'.npy'), ksi_metrics)

    np.save(os.path.join(main_path, 'im_metrics_len'\
                         + str(inital_length) + 'states' + \
                         str(no_states) + '.npy'), im_metrics)

    np.save(os.path.join(main_path, 'fischer_polar_len' \
                         + str(inital_length) + 'states' + \
                         str(no_states) + '.npy'), fischer_polar)

    np.save(os.path.join(main_path, 'fixed_states_len' \
                         + str(inital_length) + '_states' + \
                         str(no_states) + '.npy'), fixed_states)



