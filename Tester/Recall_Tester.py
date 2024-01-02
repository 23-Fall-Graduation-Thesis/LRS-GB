def Get_test_results_single(image_datasets, dataloaders, model, f_size=2048):
    gallery_eq_query = True
    feature_size = f_size

    query_path = image_datasets.imgs
    query_label = get_id(query_path)
    query_label = np.asarray(query_label)
    gallery_label = np.asarray(query_label)

    # Change to test mode
    model = model.train(False)

    # Extract feature
    recall_ks = []
    query_feature = extract_feature(model, dataloaders, feature_size)

    sim_mat = pairwise_similarity(query_feature)
    # sim_mat = pairwise_similarity(query_feature, gallery_feature)

    if gallery_eq_query is True:
        sim_mat = sim_mat - torch.eye(sim_mat.size(0))

    recall_ks.append(Recall_at_ks(sim_mat, query_ids=query_label, gallery_ids=gallery_label, data='cub'))
    # print('{:.4f}'.format(recall_ks[0]))

    return recall_ks[0]