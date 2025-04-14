    try:
        print("calculating pseudo-bulk data")
        pdata_500 = get_pseudo_bulk_data(adata, group_by, pseudo_bulk="pseudo_bulk_500")
        try:
            print("calculating deseq2 scores")
            deseq2_results_500 = deseq2(pdata_500, group_by, reference)
        except:
            deseq2_results_500 = dict()
            
        try:
            print("calculating edgeR scores")
            edgeR_results_500 = edgeR(pdata_500, group_by, reference)
        except:
            edgeR_results_500 = dict()  
            
    except:
        print("pseudo-bulk failed")
        pdata_500 = None
        deseq2_results_500 = dict()
        edgeR_results_500 = dict()