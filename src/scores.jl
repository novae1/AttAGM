using ScikitLearn
@sk_import metrics: classification_report
@sk_import metrics: f1_score
@sk_import metrics: normalized_mutual_info_score
@sk_import metrics: adjusted_rand_score
"""
Compute avg F1 score according to paper
https://cs.stanford.edu/people/jure/pubs/agmfit-icdm12.pdf
"""

function avg_score(true_matrix,predicted_matrix,score_fun=f1_score)
    N , K_pred = size(predicted_matrix)
    K_true = size(true_matrix)[2]
    F_g = 0
    for i in 1:K_true
        best_matching = maximum([ score_fun(
                                    true_matrix[:,i],
                                    predicted_matrix[:,j]
                                    ) 
                                for j in 1:K_pred])
        F_g += best_matching
    end
    F_g /= K_true

    F_d = 0
    for i in 1:K_pred
        best_matching = maximum([ score_fun(
                                    true_matrix[:,j],
                                    predicted_matrix[:,i]
                                    ) 
                                for j in 1:K_true])
        F_d += best_matching
    end
    F_d /= K_pred
    return 0.5 * (F_d + F_g)
end

function compute_all_metrics(true_matrix,predicted_matrix)
    return [avg_score(true_matrix,predicted_matrix,f1_score),
            avg_score(true_matrix,predicted_matrix,normalized_mutual_info_score),
            avg_score(true_matrix,predicted_matrix,adjusted_rand_score)]

end