import numpy as np
import scipy.stats as stats


# =============================================================================
# Traitement des valeurs aberrantes
# =============================================================================

def mean_pixels(img, pix):
    '''Calcule la moyenne des pixels voisins'''
    dim = img.shape[0]
    mean = - img[pix]
    for i in range(-1,2):
        for j in range(-1,2):
            mean += img[(pix[0] + i) % dim, (pix[1] + j) % dim]
    mean = mean / 8
    return mean

def outlier_replace(img, crit):
    '''Remplace les valeurs aberrantes dans l'image img
        (valeurs supérieures au critère crit) par la moyenne
        des pixels voisins'''
    index = np.where(img >= crit)
    for i in range(len(index[0])):
        img[index[0][i], index[1][i]] = mean_pixels(img, (index[0][i], index[1][i]))
    return

def Grubbs_critical_value(size, alpha):
    '''Calcule le valeur critique pour le test de Grubbs'''
    t_dist = stats.t.ppf(1 - alpha / (size), size - 2)
    numerator = (size - 1) * np.sqrt(np.square(t_dist))
    denominator = np.sqrt(size) * np.sqrt(size - 2 + np.square(t_dist))
    critical_value = numerator / denominator
    return (critical_value)

def Grubbs_test(img, G_crit):
    '''Vérifie si la valeur de Grubbs de l'image est inférieure 
    à la valeur critique G_crit'''
    mean = np.mean(img)
    max_dev = np.max(img) - mean
    s = np.std(img)
    G = max_dev / s
    #print(G, G_crit)
    return(G < G_crit)

def Grubbs_method(img, G_crit):
    '''Effectue le remplacement des valeurs aberrantes de l'image selon le 
    test de Grubbs'''
    while ~Grubbs_test(img, G_crit):
        outlier_replace(img, np.max(img))
    return

def Grubbs_data(Set):
    '''Effectue le remplacement des valeurs aberrantes du jeu de données Set 
    selon le test de Grubbs'''
    shape = np.shape(Set)
    G_crit = Grubbs_critical_value(shape[1]**2, 0.01)
    Grubbs_method(Set, G_crit)
    return(Set)

# =============================================================================
# Normalisation
# =============================================================================

def norm(img):
    '''Retourne une image normalisée (ici norme minmax)'''
    norm_img = (img - np.min(img)) / (np.max(img) - np.min(img))
    return (norm_img)

def norm_data(Set):
    '''Normalise un jeu de donées (normalisation image par image)'''
    shape = np.shape(Set)
    Set_out = np.zeros(shape)
    Set_out = norm(Set)
    return(Set_out)

def global_norm(Set):
    '''Normalise un jeu de données (normalisation sur le set entier)''' 
    shape = np.shape(Set)
    n = shape[-1]
    Set_out = np.zeros(shape)
    for i in range(n):
        Set_out[:,:,:,i] = norm(Set[:,:,:,i])
    return(Set_out)
