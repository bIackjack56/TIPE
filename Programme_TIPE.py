
import numpy as np
import matplotlib.pyplot as mplt
from math import exp, sqrt, atan
import random
import scipy.optimize as sco
import networkx as nx
import scipy.stats as sst
import inspect
import copy
import pickle

###Modele deterministe

#Lecture des fichiers du gouvernement : jeu de donees du covid


def lecture_fichier(nom_fichier):
    """fonctions renvoyant la liste des cas confirmés par jour"""
    fichier = open(nom_fichier, 'r')
    cas_confirm = np.array([], dtype=int)
    for ligne in fichier:
        valeur = ligne.split(',')[14]
        if valeur == '':
            valeur = 0
        cas_confirm = np.append(cas_confirm,int(valeur))
    n = len(cas_confirm)
    yy = []
    for i in range( n-7):
        c = 0
        for j in range(7):
            c += cas_confirm[i + j]
        yy.append(c/7)
    fichier.close()
    return yy

def graphique_donne(array):
    """fonction créant un graphique avec la liste passée en paramètre lissée sur 7 jours"""
    x = [i for i in range(len(array))]
    mplt.plot(x,array)
    mplt.show()


graphique_donnees_covid_gouv = lecture_fichier("C:\\Users\\jluca\Documents\\TIPE\\Programmes_et_fichier\\donnees-covid-modele-sir-gouv.csv")

#Modele SIR

def I_SIR(t,N,alpha, beta) :
    d = beta*N - alpha
    #Equation du modèle linéarisé
    return exp((d) * (t))

def I_prim_SIR(t,N, alpha,beta):
    d = beta*N - alpha
    return d* exp(d*t)


def determination_para(deb, fin):
    """regression exponentielle des données du gouvernement"""
    xdata = np.arange(deb, fin)
    ydata = np.array(graphique_donnees_covid_gouv[deb:fin])
    def expo(x,b,c) :
        return b*np.exp(b*x) + c
    p, _ = sco.curve_fit(expo,xdata,ydata)
    return p

def graphique_SIR(N, deb, fin):
    x = [i for i in range(400)]
    n = 400
    y = []
    i = 0
    parametre = determination_para(deb, fin)
    b,c = parametre[0], parametre[1]
    alpha = 0.29
    beta = (b + alpha) / N
    while i < n:
        y.append(I_prim_SIR(i, N, alpha, beta )) #valeur correspondant le mieux
        i += 1
    mplt.plot(x,y)
    mplt.axis([0,400,0,60000])
    graphique_donne(graphique_donnees_covid_gouv)
    mplt.show()


#Modele SIR resolution numérique

def para(N):
    bet = (5e-2 + al_SIR_num)/N
    return bet

#Parametres

al_SIR_num = 0.5
N_SIR_num = 10000
bet_SIR_num = 0.01
#para(N_SIR_num)


def E1(t, s, i):
   return  -bet_SIR_num*i*s
def E2(t,s,i):
    return bet_SIR_num*i*s - al_SIR_num*i

def euler_s(F1 , F2 , t0 , tf , x0 , y0 , h ):
    """ résolution du système d ’ équations différentielles x’= F (t ,x , y )
    et y’= G (t ,x , y ) par méthode d’Euler avec un pas h """
    temps = np.arange ( t0 , tf + h , h )
    S = np.zeros(len( temps ))
    I = np.zeros(len( temps ))
    S[0] = x0
    I[0] = y0
    for i in range (1 , len( temps )):
        S[ i ] = S[i - 1] + h * F1(temps[ i - 1], S[ i - 1], I[ i - 1])
        I[ i ] = I[i - 1] + h * F2(temps[ i - 1], S[ i - 1], I[ i - 1])
    return temps,S,I

def graphique_SIR_num():
    x, S,I = euler_s(E1,E2, 0, 50, N_SIR_num-1, 1, 0.1 )
    mplt.plot(x[::5], I[::5])
    rapport = N_SIR_num/65e6
    #graphique_donne(np.array(graphique_donnees_covid_gouv[50:350]) * rapport)
    mplt.show()

#Chaine de Markov

#alpha : probabilité que une personne en infecte une autre
#beta : probabilité qu'une personne guérisse

def taux_inf_lim(beta, pop):
    return (1-beta)/pop

def beta_max(taux_inf,pop):
    return 1 - pop*taux_inf


def mat_stochastique(beta, taux_inf, n):
    """crée la matrice stochastique de la chaine de markov pour une population de taille n
    donc la matrice est de taille n+1
    attention, on doit avoir n*taux_inf < 1 - beta"""
    L = [0 for i in range(n+1)]
    L[0] = 1
    mat = [L]
    for i in range(1, n):
        alpha = taux_inf * i
        L =[0 for j in range(i-1)]
        L.extend([beta , 1 - beta - alpha, alpha])
        L.extend([0 for k in range(n-i-1)])
        mat.append(L)
    L = [0 for i in range(n-1)]
    L.append(beta)
    L.append(1-beta)
    mat.append(L)
    m = np.array(mat)
    print(est_sto_reg(m))
    return m


def est_sto_reg(matrice):
    """renvoie true si la matrice est stochastique"""
    for i in range(len(matrice[0])):
        L = 0
        for j in range(len(matrice[0])):
            y = matrice[i][j]
            if abs(y) > 1 :
                return False
            if y < 0 :
                return False
            L += y
        if abs(1-L)>1e-8:
            return False, i
    return True


def matrice_puissance(matrice, etape):
    """eleve la matrice à la puissance étape+1"""
    if etape == 1 :
        return matrice
    else:
        if not(est_sto_reg(matrice)):
            print("PB")
        else :
            m = matrice
            for k in range(1,etape + 1):
                m = m.dot(matrice)
            return m


def passage_etat(etape, matrice_stochastique):
    """renvoie la loi de probabilité pour passer de 1 à i
    pour i sur [0, N] en n étape"""
    M_n = matrice_puissance(matrice_stochastique, etape)
    L1 = M_n[1]
    return L1


def graphe_etape_par_etape(etape,matrice_sto):
    m = matrice_sto
    n = len(m)
    x = np.arange(0, n)
    c = ['b', 'g', 'k', 'y', 'm', 'r', 'c', 'brown' , 'summer','rgb']
    for i in range(1,etape+1):
        if i % 10 == 0 :
            mplt.bar(x,m[1], color = c[(i-1)%10] )
            mplt.yscale('log')
            mplt.axis([0, 100, 0, 1])
        m = m.dot(matrice_sto)
    mplt.show()


def graphe_repartition(etape, matrice_sto):
    """en echelle log : renvoie la probabilité d'être dans l'état k au bout de etape etapes"""
    taille_pop = len(matrice_sto[0]) - 1
    rep = passage_etat(etape, matrice_sto)
    x = np.arange(0, taille_pop + 1)
    mplt.bar(x,rep, color = 'r')
    mplt.yscale('log')

    mplt.show()
    return rep

def graphe_repartition_beta_taux_lim():
    """renvoie un graphe qui montre la répartition de probabilité des états suivant
    la valeur du taux_inf : le taux d'infection"""
    L = []
    for j in range(10):
        L.append(j*0.01)
    c = ['r', 'b', 'g', 'k', 'y', 'c', 'm']
    for i in range(10):
        m = mat_stochastique(L[i], taux_inf_lim(L[i], 100), 100)
        rep = passage_etat(100, m)
        x = np.arange(0,  101)
        l = 'beta = ' + str(L[i])
        mplt.bar(x,rep, label = l)
        mplt.axis([0, 100, 0, 0.2])
        mplt.yscale('log')
    mplt.legend()
    mplt.show()
    return rep


def calcul_chaines_nouveau(beta, b, gamma, N_init, I_init, nb_etapes):
    """calcule les valeurs des différents compartiment modélisés
    par des chaines de Markov avec des probas:
    parcours la chaine de Markov aléatoirement"""
    delta_t = 0.04
    N = [N_init]
    I = [I_init]
    M = [0]
    S = [N[0] - I[0]]
    deltat = 0.005
    for k in range(nb_etapes):
        prob_i = (beta * S[k] * I[k] / N[k]) *deltat
        prob_mg = (beta + gamma) * I[k] * deltat
        prob_inchange = 1 - prob_i - prob_mg
        prob_inco = b / (b + gamma)
        tirage = random()
        if tirage < prob_i:
            I.append(I[k] + 1)
            S.append(S[k] - 1)
            M.append(M[k])
            N.append(N[k])
        elif tirage < prob_mg + prob_i:
            I.append(I[k] - 1)
            tirage2 = random()
            # probabilité qu'un individu meurt sachant qu'il n'est plus infecté
            if tirage < prob_inco:
                # cf probas composées
                S.append(S[k] - 1)
                M.append(M[k] + 1)
                N.append(N[k] - 1)
            else:
                S.append(S[k] + 1)
                M.append(M[k])
                N.append(N[k])
        else:
            I.append(I[k])
            S.append(S[k])
            M.append(M[k])
            N.append(N[k])
    return I, S, M, N

"""for i in range(1000):
    I,S,M,N = calcul_chaines_nouveau(0.001*i, 0.01, 0.1, 100, 3, 1000)
    T = np.arange(1001)
    mplt.plot(T,N)"""

def graphique_parcours_Markov():
    I,S,M,N = calcul_chaines_nouveau(1, 1, 0.01, 100, 3, 1000)
    T = np.arange(1001)
    mplt.plot(T,N,"b", label = "population")
    mplt.plot(T,S,"r", label = "sains")
    mplt.plot(T,M, "k", label = 'mort')
    mplt.plot(T, I, "c", label = 'infectés')
    mplt.legend()
    mplt.show()


#Modèle SEIR

def coefficients(Alpha):
    """Fonction qui permet de renvoyer les paramètres du modele avec un coefficient alpha arbitraire, des coefficient beta et delta déterminés grâce aux données du gouvernement disponibles au 15 mai 2021"""
    alpha = Alpha
    beta = 1 / 14
    delta = 1.9 * 10 ** -2
    """1.9 * 10 ** -2"""
    """435 jours depuis le début de l'épidémie ; 5 713 917 cas confirmés pour 110 091 décès
    cf
    https://www.santepubliquefrance.fr/dossiers/coronavirus-covid-19/coronavirus-chiffres-cles-et-evolution-de-la-covid-19-en-france-et-dans-le-monde"""
    gamma = 0.089
    """0.089"""
    """5 116 705 guérisons
    cf
    https://fr.statista.com/statistiques/1099816/guerisons-coronavirus-monde/"""
    g_delta = (beta + delta + gamma)**2 - 4*beta*(delta + gamma - alpha)
    l4 = -((beta + gamma + delta) - sqrt(g_delta))*(1/2)
    l5 = -((beta + gamma + delta) + sqrt(g_delta))*(1/2)
    return (alpha, beta, delta, gamma, g_delta, l4, l5)

#alpha, beta, delta, gamma, g_delta, l4, l5 = coefficients(alphainit)

#Fonctions auxiliaires pour l'écriture des équations
def modele(parametre):
    """équations du modele"""
    alpha,beta, delta, gamma, g_delta, l4, l5 = parametre


    #Equations du modèle
    #i est le nombre d'infectés initiaux, c celui de contaminés initiaux
    def I_SEIR(t, i, c):
        """infecté en fonction du temps, cf diagonalisation matrice"""
        def X_SEIR(i, c):
            return ((beta + l4)/(l5-l4)) * (-i + c * (beta + l5) / alpha)
        def Y_SEIR(i, c):
            return ((beta + l5)/(l4-l5)) * (-i + c * (beta + l4) / alpha)

        return X_SEIR(i, c)*np.exp(l4 * t) + Y_SEIR(i, c)* np.exp(l5 * t)

    def C_SEIR(t, i, c):
        """contaminés en fonction du temps"""
        return (1 / (l5 - l4)) * ( - alpha * i + c * (beta + l5)) * np.exp(l4 * t) + (1 / (l4 - l5)) * ( - alpha * i + c * (beta + l4)) * np.exp(l5 * t)

    def S_SEIR(N, t, i, c):
        E = - i / (l4 * (l5 - l4)) + c * (beta + l5) / (alpha * l4 * (l5 - l4))
        F = - i / (l5 * (l4 - l5)) + c * (beta + l4) / (alpha * l5 * (l4 - l5))
        s = N + (beta + l4) * alpha * E * (1 - np.exp(l4 * t)) + (beta + l5) * alpha * F * (1 - np.exp(l5 * t))
        return s

    def M_SEIR(t, i, c, m):
        E = - i / (l4 * (l5 - l4)) + c * (beta + l5) / (alpha * l4 * (l5 - l4))
        F = - i / (l5 * (l4 - l5)) + c * (beta + l4) / (alpha * l5 * (l4 - l5))
        return delta * (beta + l4) * E * (np.exp(l4 * t) - 1) + delta * (beta + l5) * F * (np.exp(l5 * t) - 1) + m

    def G_SEIR(t, i, c, g):
        E = - i / (l4 * (l5 - l4)) + c * (beta + l5) / (alpha * l4 * (l5 - l4))
        F = - i / (l5 * (l4 - l5)) + c * (beta + l4) / (alpha * l5 * (l4 - l5))
        return gamma * (beta + l4) * E * (np.exp(l4 * t) - 1) + gamma * (beta + l5) * F * (np.exp(l5 * t) - 1) + g

    def somme_SEIR(t, i, c):
        N = 65 * 10 ** 6
        return I_SEIR(t, i, c) + C_SEIR(t, i, c) + S_SEIR(N, t, i, c) + G_SEIR(t, i, c) + M_SEIR(t, i, c)

    return  I_SEIR, C_SEIR, M_SEIR, G_SEIR, somme_SEIR


def graphique_SEIR(al, deb, fin, i_ini, c_ini):
    """créer un graphique pour une valeur de alpha entre les dates deb et fin"""
    parametres = coefficients(al)
    I,C, _,_,_ = modele(parametres)
    x = np.arange(0, fin-deb)
    y = I(x,i_ini, c_ini)
    xx = np.arange(deb,fin)
    return y, I(fin-deb, i_ini, c_ini), C(fin-deb, i_ini, c_ini)



def graphe_SEIR_confinement():
    """créer un graphe avec les valeurs de alpha qui varient dans le temps"""
    division_temps = [ (0,220), (220, 265), (265,350), (350, 380), (380, 400)]
    valeur_alpha_associés = [0.29, 0.029, 0.1, 0.193, 0.029]
    n = len(division_temps)
    inf_ini = 1
    c_ini = 0
    xx = np.arange(0,400)
    yy = graphique_donnees_covid_gouv[0:400]
    y = []
    x = []
    for i in range(n):
        print(inf_ini, c_ini)
        deb, fin = division_temps[i]
        ydata, inf_ini, c_ini = graphique_SEIR(valeur_alpha_associés[i], deb,fin, inf_ini, c_ini)
        y.append(ydata)
        x.append([i-10 for i in range(deb,fin)])


    for j in range(len(x)):
        mplt.plot(x[j], y[j], 'b')
    mplt.plot(xx[0:400],yy,'m')
    mplt.axis([0,400, 0, 100000])
    mplt.show()
    return
()
###Modeles Stochastiques



#Premiere idée

probabilite = { 'inf': 0.7 , 'guerison' : 0.4,  'deces' : 0.005  }

def tableau_premid(taille):
    """crée un tableau de 0 de coté taille"""
    tab = []
    for i in range(taille):
        L = []
        for j in range(taille):
            L.append(0)
        tab.append(L)
    return np.array(tab)

def affichage_tab_premid(tab):
    n= len(tab)
    mortx = []
    morty = []
    sainx = []
    sainy = []
    infectex = []
    infectey = []
    for i in range(n):
        for j in range(n):
            if tab[i][j] == 0:
                sainx.append(i)
                sainy.append(j)
            elif tab[i][j] == 1 :
                infectex.append(i)
                infectey.append(j)
            elif tab[i][j] == 8 :
                mortx.append(i)
                morty.append(j)
    mplt.scatter(sainx,sainy, c = 'b')
    mplt.scatter(infectex, infectey, c = 'r')
    mplt.scatter(mortx, morty, c = 'k')
    mplt.show()



#fonctions modifiant l'état d'une case
def mort_premid(matrice, i, j):
    matrice[i,j] = 8

def infection_premid(matrice, i, j):
    matrice[i,j] = 1

def guerison_premid(matrice, i, j):
    matrice[i,j] = 0

def coordone_voisin_premid(n, i, j):
    return [((i-1)%n,j), (i, (j+1)%n), ((i+1)%n, j), (i, (j-1)%n)]


def infection_voisin_premid(matrice, i_m, j_m):
    """infecte le voisin de la case de coordonnées i_m, j_m, suivant la proba
    definie plus haut"""
    n = len(matrice)
    proba_inf = probabilite['inf']
    coordo_voisin = coordone_voisin_premid(n, i_m, j_m) #liste avec coordonnées des 4 voisins
    nouv_infecte = [] #voisins infecte par virus
    for i in range(4):
        prob = randint(0,100)
        if prob < proba_inf*100:
            a,b = coordo_voisin[i]
            a, b = int(a), int(b)
            if matrice[a][b] == 0: #on infecte si voisin pas deja infecté
                infection_premid(matrice, a, b)
                nouv_infecte.append((a, b))

    return nouv_infecte #on renvoit la liste des voisins ayant été infectés



def tour_infection_premid(matrice, liste_infecte):
    """prends la liste de tout les infecter et infecte ou non leur voisins"""
    n = len(liste_infecte)
    infecte2 = liste_infecte.copy()
    nb_inf = 0
    for i in range(n):
        a, b = liste_infecte[i]
        nouv = infection_voisin_premid(matrice,a,b)
        infecte2.extend(nouv)
        nb_inf += len(nouv)
    return infecte2, nb_inf #renvoie la liste de tout les infectés de la population
#et le nombre d'infectés en plus


def desinfection_premid(matrice, liste_infecte):
    """desinfecte une case selon la probabilité definie plus haut"""
    n = len(liste_infecte)
    infecte2 = []
    nb_desinf = 0
    proba = probabilite['guerison']
    for i in range(n):
        p = randint(0,100)
        if p < proba*100:
            a, b = liste_infecte[i]
            guerison_premid(matrice, a, b)
            nb_desinf += 1
        else:
            infecte2.append(liste_infecte[i]) #on copie la liste des infectes
            #en ne gardant que les infectes et pas les gueris
    return infecte2, nb_desinf #renvoie la liste des infectes et le nombre de gueris

def tour_mort_premid(matrice, liste_infecte):
    """tue les personnes selon la probabilité definie plus haut"""
    n = len(liste_infecte)
    infecte2 = []
    nb_mort = 0
    proba = probabilite['deces']
    for i in range(n):
        p = randint(0,100)
        if p < proba * 100:
            a,b = liste_infecte[i]
            mort_premid(matrice, a, b)
            nb_mort += 1 #on compte le nombre de mort
        else:
            infecte2.append(liste_infecte[i])#on garde les infectes restant
    return infecte2, nb_mort #renvoie la liste des infectes et le nombre de mort

def epidemie_premid(tours, cote_tab, inf_initial):
    """genere l'epidemie, avec nombre tour, et liste infectes initiaux sous forme d'une liste des coordonnées des individu dans le tableau"""
    t = tableau_premid(cote_tab)
    affichage_tab_premid(t)
    infecte = inf_initial
    nb_inf = len(infecte)
    total_inf = [nb_inf]

    deces = [0]
    cumul_deces = 0

    guerie = [0]
    cumul_guerie = 0

    for i in range(nb_inf): #on place la liste des infectes initiaux
    #dans un état infecté
        a,b = inf_initial[i]
        infection_premid(t, a, b)
    for j in range(tours):
    #on simule l'émidémie
        if j % 10 == 0:
            affichage_tab_premid(t)
        #time.sleep(0.5)
        infecte, nb = tour_infection_premid(t, infecte)
        infecte, nb_mort = tour_mort_premid(t, infecte) #on tue
        infecte, nb_desin = desinfection_premid(t, infecte)#on guérie
        #on infecte

        nb_inf -= nb_mort #on enleve les morts au nb d'infecte
        nb_inf -= nb_desin#on enleve les guéries
        nb_inf += nb #on rajoute les nouveaux infectes

        cumul_deces += nb_mort #on compte les mort totaux
        total_inf.append(nb_inf) #on crée une liste des infectes totaux
        deces.append(cumul_deces)#on crée une liste des morts totaux

        cumul_guerie += nb_desin
        guerie.append(cumul_guerie)


        #print(nbre_infectes[i])
    return total_inf, deces,guerie, t #, guerie
    #on renvoie les deux listes des infectés et mort totaux


def graphique_STO_prem_id(tour, cote_tab):
    """crée le graphe de l'épidémie"""
    x = np.arange(0,tour+1)
    I, D,G ,t = epidemie(tour, cote_tab, [(0,0), (1,1), (2,2)])
    mplt.axis([0,tour, 0, 100])

    mplt.plot(x,I, 'r')
    mplt.plot(x, D, 'k')
    #mplt.plot(x, G, 'b')
    mplt.show()






###Graphes et réseaux de contact


#Modèle simple : complexification de l'idée naïve :

p_croisée = 50

#Fonctions utile au modèle

def arrete_to_adj(tab, S):
    """renvoie le tableau d'adjacence correspondant à la liste d'arretes"""
    t = list(tab.edges)
    n = len(t)
    L = [[] for i in range(S)]
    for el in t:
        s1,s2 = el
        L[s2].append(s1)
        L[s1].append(s2)
    return L

def affiche_pop(tab):
    """affiche le graphe représenté par le tableau d'adjacence tab"""
    tab_adj = tab.copy()
    n = len(tab_adj)
    G = []
    mat = []
    for k in range(n):
        mat.append([False for i in range(n)])
    print(mat)
    for i in range(n):
        m = len(tab_adj[i])
        for j in range(m):
            if not mat[i][j] :
                G.append((i,j))
                mat[i][j] = True
                mat[j][i] = False
    H = nx.Graph()
    H.add_edges_from(G)
    nx.draw_circular(H)
    plt.show()



def lissage(liste):
    """fonction qui lisse les données sur 7 jours"""
    n = len(liste)
    L= []
    for i in range(n-7):
        c = 0
        for j in range(7):
            c += liste[i + j]
        L.append(c / 7)
    return L


#paramètre de guérison-infection-mort
coefs = (0.004,0.002)#coefficient d'infection différent selon le graphe
coeff_g = 0.15
coef_m = 0.0001


#coef infection est la proba d'infection correspondant au graphe (SW ou BA)
#liste infecté est la liste des sommets deja infectés
def infection_bas(coef_infection, graphe, liste_infecte):
    n = len(liste_infecte)
    copie = copy.deepcopy(liste_infecte)
    for i in range(n):
        if liste_infecte[i] == 1:
            voisin = graphe[i]
            for el in voisin:
                if  copie[el] == 0 and random.random() < coef_infection:
                    copie[el] = 1
    return copie


#l'état de la population est modélisé par une liste, dont
#chaque opérande correspond à l'état d'une personne :
# 0 pour sain, 1 pour infecté, 2 pour mort.
#coef_inf, coef_guerison, coef_mort sont les probas respectivement
#d'inféction, de guérison, et de mort correspondant au graphe
#g_SW est le graphe de watt strogzard
#l_inf_prec est la liste des infectés issuent de la journée anterieure

def guerison_bas(coef_guerison, liste_infecte):
    """détermine les personne guéri ou non dans la journée : les personnes infectées guérissent avec la probabilité coef_guerison."""
    n = len(liste_infecte)
    copie = copy.deepcopy(liste_infecte)
    for i in range(n):
        if copie[i] == 1 and random.random() < coef_guerison:
            copie[i] = 0
    return copie

def mort_bas(coef_mort, liste_inf):
    """détermine les personne guéri ou non dans la journée, les personnes meurent avec la probabilité
    coef_mort"""
    n = len(liste_inf)
    copie = copy.deepcopy(liste_inf)
    compteur = 0
    for i in range(n):
        if copie[i] == 1 and random.random() < coef_mort:
            copie[i] = 2
            compteur += 1
    return copie, compteur



def journee_bas(coef_infs, coef_guerison, coef_mort, g_SW, g_BA, l_inf_prec):
    """Renvoie l'état de la population après une 'journée' sous forme de liste (ie. une fois que chaque personne
    à eu la possibilité de changer d'état  """
    n = len(l_inf_prec)
    cSW,cBA = coef_infs
    m = infection_bas(cSW, g_SW, l_inf_prec)
    p = infection_bas(cBA, g_BA, l_inf_prec)
    for i in range(n):
       if m[i] != p[i]:
           m[i] = 1
    liste_mort, compteur = mort_bas(coef_mort, m)
    return guerison_bas(coef_guerison, liste_mort ), compteur

def simulation_bas(t_pop):
    """effectue une simulation de l'épidémie sur une population de taille t_pop"""
    gWS =  arrete_to_adj(nx.connected_watts_strogatz_graph(t_pop, 6, 0.3), t_pop)
    gBA = arrete_to_adj(nx.barabasi_albert_graph(t_pop, p_croisée), t_pop)
    l_inf = [(0)] * t_pop
    l_inf[0] = 1
    nb_inf = []
    nb_mort = []
    j = []
    for i in range(100):
        print(i, "\n")
        l_inf, compte_mort = journee_bas(coefs, coeff_g, coef_m, gWS, gBA, l_inf)
        somme_inf = 0
        for k in range(len(l_inf)):
            if l_inf[k] == 1:
                somme_inf += 1
        nb_inf.append(somme_inf)
        nb_mort.append(compte_mort)
        j.append(i)
    return l_inf, nb_inf, j, nb_mort

def graphique_resaux_basique():
    s, inf,j, mort = simulation_bas(2600)
    l = 'coefs_inf = ' +  str(coefs) + ' , coeff_g = ' + str(coeff_g) + ' , coeff_mort = ' + str(coef_m)
    mplt.plot(j,inf, label = l)
    mplt.legend()
    mplt.show()
    mplt.plot(j[:len(j)-7], lissage(mort), label = l)
    mplt.legend()
    mplt.show()



#Ajout d'un cluster dans la population : ie un grand nombre de personne devient infectée


#si les gens sont des inconnus
def cluster_inconnu(taille, t_pop, list_inf):
    """choisie aléatoirement des individus de la population, et si ils ne sont pas mort ou déja infectés, les infectes -> représente un cluster ou les individu on très peu de liens entre eux, type concert"""
    i = 0
    c = 0
    while i < taille-1 and c < t_pop-1:
        k = random.randint(1,t_pop-1)
        if not(list_inf[k] == 1 or list_inf[k] == 2):
            list_inf[k] = 1
            i += 1
        c += 1
    return list_inf



def simulation_avec_cluster_inco(t_pop, j_clust, taille_clu):
    """effectue une simulation de l'épidémie sur une population de taille t_pop, avec apparition d'un cluster au jour j_clust"""
    gWS =  arrete_to_adj(nx.connected_watts_strogatz_graph(t_pop, 6, 0.3), t_pop)
    g_BA = arrete_to_adj(nx.barabasi_albert_graph(t_pop, p_croisée), t_pop)#on ne crée pas un graphe
    #a chaque boucle sinon trop long
    l_inf = [0] * t_pop
    l_inf[0] = 1
    nb_inf = []
    nb_mort = []
    j = []
    for i in range(300):
        print(i, "\n")
        l_inf, compte_mort = journee_bas(coefs, coeff_g, coef_m, gWS, g_BA, l_inf)
        somme_inf = 0
        somme_mort = 0
        for k in range(len(l_inf)):
            if l_inf[k] == 1:
                somme_inf += 1
            elif l_inf[k] == 2:
                somme_mort += 1
        nb_inf.append(somme_inf)
        nb_mort.append(somme_mort)
        if i == j_clust:
            cluster_inconnu(taille_clu, t_pop, l_inf)
        j.append(i)
    return l_inf, nb_inf, j, nb_mort


def graphique_reseaux_cluster_inco():
    s, inf,j, mort = simulation_avec_cluster_inco(2600, 100, 500)
    mplt.plot(j[:len(j)-7], lissage(mort))
    mplt.plot(j,inf)
    mplt.show()

#si les gens se connaissent

def cluster_connu(distance, i, t_pop, liste_inf, gWS):
    if distance == 0:
        liste_inf[i] = 1
        return
    else :
        n = len(gWS[i])
        for j in range(n):
            liste_inf[gWS[i][j]] = 1
            cluster_connu(distance-1,j, t_pop, liste_inf, gWS )

def simulation_avec_cluster_co(t_pop, j_clust, taille_clu):
    """effectue une simulation de l'épidémie sur une population de taille t_pop, avec apparition d'un cluster au jour j_clust"""
    gWS =  arrete_to_adj(nx.connected_watts_strogatz_graph(t_pop, 6, 0.3), t_pop)
    gBA = arrete_to_adj(nx.barabasi_albert_graph(t_pop, p_croisée), t_pop)#on ne crée pas un graphe
    #a chaque boucle sinon trop long
    l_inf = [0] * t_pop
    l_inf[0] = 1
    nb_inf = []
    nb_mort = []
    j = []
    for i in range(300):
        print(i, "\n")
        l_inf,compte_mort = journee_bas(coefs, coeff_g, coef_m, gWS, gBA, l_inf)
        somme_inf = 0
        somme_mort = 0
        for k in range(len(l_inf)):
            if l_inf[k] == 1:
                somme_inf += 1
            elif l_inf[k] == 2:
                somme_mort += 1
        nb_inf.append(somme_inf)
        nb_mort.append(somme_mort)
        if i == j_clust:
            k = random.randint(0, t_pop-1)
            cluster_connu(taille_clu, k, t_pop, l_inf, gWS)
        j.append(i)
    return l_inf, nb_inf, j, nb_mort


def graphique_reseaux_cluster_co():
    s, inf,j, mort = simulation_avec_cluster_co(2600, 100, 4)
    mplt.plot(j[:len(j)-7], lissage(mort))
    mplt.plot(j,inf)
    mplt.show()


def graphique_comparaison():


    x = [i for i in range(400)]
    xx = [i/8.5 for i in range(400)]
    n = 400
    y = []
    i = 0
    alpha = 0.29/3
    b = 5e-2/3

    beta = ((b + alpha)/3000)
    for i in range(n):
        y.append(I_SIR(x[i], 3000, alpha, beta )) #valeur correspondant le mieux
    mplt.plot(xx,y)
    #mplt.axis([0,400,0,60000])

    #graphique_donne(graphique_donnees_covid_gouv)


    s, inf,j, mort = simulation_bas(3000)
    l = 'coefs_inf = ' +  str(coefs) + ' , coeff_g = ' + str(coeff_g) + ' , coeff_mort = ' + str(coef_m)
    mplt.plot(j,inf, label = l)
    #mplt.legend()
    mplt.show()
    #mplt.plot(j[:len(j)-7], lissage(mort), label = l)
    #mplt.legend()





###Modèle avec temps d'infection fixé


#Variables gloables
p_croisee_IF = 10
compteur_jour_IF = 0
dure_inf_IF = 10

#coef infection est la proba d'infection correspondant aux différents graphes
#liste infecté est la liste des sommets deja infectés
def infection_IF(coef_infection, graphe, liste_infecte, compteur_jour):
    n = len(graphe)
    copie = copy.deepcopy(liste_infecte)
    compteur_infecte = 0
    for i in range(n):
        etat, jour_inf = liste_infecte[i]
        if etat == 1:
            voisin = graphe[i]
            for el in voisin:
                etatv, jour_infv = copie[el]
                if   etatv == 0 and random.random() < coef_infection:
                    copie[el] = (1, compteur_jour)
                    compteur_infecte += 1
    return copie, compteur_infecte


#l'état de la population est modélisé par une liste
#dont chaque opérande correspond à l'état d'une personne :
# 0 pour sain, 1 pour infectée, 2 pour morte.
#coef_inf, coef_guerison, coef_mort sont les probas respectivement
#d'infection, de guerison, et de mort correspondant aux graphes
#g_SW est le graphe de Watt Strogzard
#l_inf_prec est la liste des infectés issues de la journée anterieure

def guerison_IF(coef_guerison, liste_infecte,compteur_jour):
    """détermine les personnes guéries ou non dans la journée"""
    n = len(liste_infecte)
    copie = copy.deepcopy(liste_infecte)
    compteur_guerie = 0
    for i in range(n):
        etat, jour_inf = copie[i]
        if etat == 1 and compteur_jour  > jour_inf + dure_inf_IF :
            copie[i] = (0, -1)
            compteur_guerie += 1

    return copie, compteur_guerie

def mort_IF(coef_mort, liste_inf, compteur_jour):
    """détermine les personnes mortes ou non dans la journée"""
    n = len(liste_inf)
    copie = copy.deepcopy(liste_inf)
    for i in range(n):
        etat, jour_inf = copie[i]
        if  etat == 1 and random.random() < coef_mort:
            copie[i] = (2,-1)
    return copie



def journee_IF(coef_infs, coef_guerison, coef_mort, g_SW, g_BA, l_inf_prec, compteur_jour):
    """Renvoie l'état de la population après une journée sous forme de liste """
    n = len(g_SW)
    cSW,cBA = coef_infs
    m, c1 = infection_IF(cSW, g_SW, l_inf_prec, compteur_jour)
    p, c2 = infection_IF(cBA, g_BA, l_inf_prec, compteur_jour)
    for i in range(n):
       if m[i] != p[i]:
           m[i] = (1,compteur_jour)

    copie, compteur_guerie = guerison_IF(coef_guerison, mort_IF(coef_mort, m, compteur_jour), compteur_jour)
    return copie, compteur_guerie, c1, c2, compteur_jour+1

def simulation_IF(t_pop):
    """effectue une simulation de l'épidémie sur une population de taille t_pop"""
    coefs = (0.001,0.001)
    coef_m = 0.0001
    coeff_g = 0.001#n'est pas utile, mais est pris en argument par les autres fonctions
    gWS =  arrete_to_adj(nx.connected_watts_strogatz_graph(t_pop, 6, 0.3), t_pop)
    g_BA = arrete_to_adj(nx.barabasi_albert_graph(t_pop, p_croisee), t_pop)

    l_inf = [(0,0)] * t_pop
    l_inf[0] = (1,0)
    nb_inf = []
    nb_mort = []
    nb_gueris = []
    j = []
    cas_journ1 = []
    cas_journ2 = []
    compteur_jour = 0

    for i in range(200):
        print(i, "\n")
        l_inf, compteur_guerie,c1, c2,  compteur_jour = journee_IF(coefs, coeff_g, coef_m, gWS, g_BA, l_inf, compteur_jour)

        somme_inf = 0
        somme_mort = 0
        somme_gueri = 0
        for k in range(len(l_inf)):
            etat, jour = l_inf[k]
            if etat == 1:
                somme_inf += 1
            elif etat == 2:
                somme_mort += 1
        if somme_inf == 0 : #permet de ne pas simuler inutilement
            return l_inf, nb_inf, j, nb_mort, nb_gueris, cas_journ1, cas_journ2
        cas_journ1.append(c1)
        cas_journ2.append(c2)
        nb_inf.append(somme_inf)
        nb_mort.append(somme_mort)
        nb_gueris.append(compteur_guerie)
        j.append(i)
    return l_inf, nb_inf, j, nb_mort, nb_gueris, cas_journ1, cas_journ2



def graphique_temps_remission_IF():
    s, inf,j, mort,guerie, cj1, cj2 = simulation_IF(2000)

    mplt.plot(j,inf, c = "r")
    mplt.plot(j,guerie, c = "b")
    mplt.plot(j, mort, c = "k")
    mplt.show()
    mplt.plot(j[7:],lissage(cj1), c = 'g', label = 'infecté SW' )
    mplt.plot(j[7:],lissage(cj2), c = 'm', label = 'infecte BA')
    mplt.legend()
    mplt.show()
    return inf


def pleins_simulation_IF():
    global dure_inf_IF
    moy = 0
    t= [0,2,5,10,20,30,50,100]
    moyenne = []
    for i in range(8):
        for j in range(50):
            print('SIMULATION', j)
            s, inf,j, mort,guerie, c1,c2 = simulation_IF(1000)
            moy += inf[len(inf)-1]
        moyenne.append(moy/50)
    return moyenne

def epidemie_moyenne():
    moy = [0 for i in range(200)]
    moy_par_j1 = [0 for i in range(200)]
    moy_par_j2 = [0 for i in range(200)]
    jours = [i for i in range(200)]
    for i in range(50):
            print('SIMULATION', i)
            s, inf,jo, mort,guerie, c1,c2= simulation_IF(1000)
            for j in range(len(inf)):
                moy[j] += inf[j]/50
                moy_par_j1[j] += c1[j]
                moy_par_j2[j] += c2[j]

    mplt.plot(jours, moy)
    mplt.show()
    mplt.plot(jours[7:], lissage(moy_par_j1), c = 'g')
    mplt.plot(jours[7:], lissage(moy_par_j2), c =  'm')
    mplt.show()





###Modèle individuo-centre 6 paramètres
##Fonction auxiliaire

def integrale(p, a, b):
    """prend en entré le polynome sous forme de liste, renvoie son intégrale sur a,b"""
    pp = np.poly1d(p)
    ppp = pp.integ()
    return ppp(b) - ppp(a)


def integrale_pas_poly(f,a,b):
    """renvoie l'integrale de la fonction sur a,b"""
    n = 100000
    h = (b-a)/n
    I = 0
    for i in range(n):
        I+= f(a+ i*h)* h
    return I

def arrete_to_adj(tab, S):
    """renvoie le tableau d'adjacence correspondant à la liste d'arretes"""
    t = list(tab.edges)
    n = len(t)
    L = [[] for i in range(S)]
    for el in t:
        s1,s2 = el
        L[s2].append(s1)
        L[s1].append(s2)
    return L

def decoupage_01(p):
    """renvoie la loi de probabilité de la variable aléatoire décrite par le
     polynôme suivant le découpage en tranches d'âge de 1 ans
    p est passé en paramètre pour éviter de le recalculer à chaque fois
    p est le polynôme issu de la régression polynomiale de lecture_age
    """
    tranche = [(i,i+1) for i in range(0, 110)]
    n = len(tranche)
    loi = []
    for j in range(n):
        loi.append(integrale(p, *tranche[j]))
    return loi

def echelle_proba(p):
    """aide au calcul pour plus tard : permet un découpage de l'intervalle [0,1] en
    tranche de probabilité
    p est passé en paramètre pour ne pas avoir à le recalculer"""
    Loi = decoupage_01(p)
    n = len(Loi)
    echelle_proba = [0]
    for i in range(1,n):
        echelle_proba.append(sum(Loi[:i]))
    return echelle_proba

def creer_densite(fonction):
    it = integrale(fonction, 0,110)
    return fonction * 1/it


## Initialisation de la population


##fonctions créant la répartition de probabilité de l'âge d'une personne

def lecture_age():
    """lit le fichier de repartition de la population
    française en fonction de l'age, effectue une régression polynomiale,
     et renvoie le polynome ainsi obtenue normalisé
     (divisé par l'integrale du polynome sur 0, 110)
    Source :"""
    fichier = open("C:\\Users\\jluca\\Documents\\TIPE\\programmes\\age_france.csv", "r")
    x = []
    age = []
    tant = []
    for ligne in fichier:
        l = ligne.split(",")
        #print(l)
        x.append(int(l[0]))
        age.append(int(l[3])/67e6)
    p =  np.polyfit(x,age, 10)#creer un polynome approximant la courbe pop = f(age)
    p = 1/integrale(p, 0,100) * p
    p = np.poly1d(p)

    #graphique
    #mplt.bar(x, age, label = "age normé")
    mplt.plot(x, p(x), label = "polynome")
    mplt.legend()
    #mplt.show()
    fichier.close()
    return p/integrale(p,0,110)


def age_f(p,e_p):
    """renvoie une tranche d'âge aléatoire en suivant la loi de probabilité aproximée
    valable dans la population reelle, p est le polynome de regression associé aux données
    e_p doit être calculé par echelle_probabilité(p)"""
    tirage = random.random()
    for i in range(len(e_p)):
        if tirage > e_p[i] and tirage < e_p[i+1] :
            return i
    return

## Fonction permettant de déterminer la probabilitée d'être infectée suivant l'âge

duree_fichier = 50

def lecture_inf():
    """renvoie un polynome qui approxime au mieux les données réelles du nombre d'infectés en fonction de l'âge
    dont on tire la probabilité de l'âge sachant qu'il y a eu infection
    Source : https://www.inspq.qc.ca/covid-19/donnees/age-sexe"""
    fichier = open("C:\\Users\\jluca\\Documents\\TIPE\\Programmes_et_fichier\\fichier_age_inf_quebec.csv",'r')
    fichier.readline()
    age = []
    nb_inf = []
    for ligne in fichier:
        l = ligne.split(',')
        age.append(int(l[0]))
        nb_inf.append(int(l[1]))
    p = np.poly1d(np.polyfit(age,nb_inf, 3))
    fichier.close()
    #graphique
    """l1 = 'données réelles'
    l2 = 'polynome'
    mplt.plot(age, nb_inf, label = l1)
    x = [i/2 for i in range(1,220)]
    mplt.plot(x, p(x), label = l2)
    mplt.axis([0,100, 0,200000])
    mplt.legend()
    mplt.show()"""

    return p, sum(nb_inf) #renvoie le polynome qui approxime et la somme des infectés totaux

def proba_age_sachant_infection(poly_inf, age):
    """age doit être entre 0 et 99"""
    inte = integrale(poly_inf, 0,100)
    proba = integrale(poly_inf, age, age+1)
    return proba/inte

def proba_inf(age, poly_age, poly_inf, nb_tot):
    """age est un entier entre 0 et 110
    p est le polynome d'interpolation renvoyé par lecture_inf
    nb_tot est le sum(nb_inf) de la fonction précédente
    poly_age est le polynome qui donne la probabilité d'être de l'âge âge"""
    tirage = random.random()
    p_age = poly_age(age)
    pinf = nb_tot/8e6 #probabilité d'être infecté en géneral, 8e6 = population Quebec
    p_sach = proba_age_sachant_infection(poly_inf, age)
    if p_age != 0 :
        probi =  p_sach * pinf/p_age  #formule des probabilités conditionelles :
        #p(age) donne la probabilité de age sachant infection, pinf est
        #la probabilité d'être infecté et p_age la probabilité de l'âge,
        # on obtient la probabilité d'infection sachant l'âge
        return probi / duree_fichier #renvoie la probabilité d'être infecté suivant l'âge

    else :

        return 0


###Fonctions permettant de déterminer la dure d'infection

t_inf_moy = 14

def dure_infection(age):
    """renvoie une durée d'infection aléatoire (suivant une gaussienne qui dépend de l'âge"""
    ecart_type = age/6  + 1
    dure = 0
    while dure < t_inf_moy:
        dure =  random.gauss(t_inf_moy, ecart_type)
    return dure


##"Fonctions permettant de déterminer la probabilité de décès


def lecture_deces():
    """renvoie un polynome qui approxime au mieux les données réelles du nombre de morts en fonction de l'âge
    dont on tire la probabilité de l'âge sachant qu'il y a eu infection et mort
    Source : https://www.inspq.qc.ca/covid-19/donnees/age-sexe"""
    fichier = open("C:\\Users\\jluca\\Documents\\TIPE\\Programmes_et_fichier\\deces_age_quebec.csv",'r')
    fichier.readline()
    age = []
    nb_deces = []
    for ligne in fichier:
        l = ligne.split(';')
        age.append(int(l[0]))
        nb_deces.append(int(l[1]))
    p = np.poly1d(np.polyfit(age,nb_deces, 7))
    fichier.close()
    #graphique
    """l1 = 'données réelles'
    l2 = 'polynome'

    mplt.plot(age, nb_deces, label = l1)
    x = [i/2 for i in range(1,220)]
    mplt.plot(x, p(x), label = l2)
    mplt.axis([0,100, 0,6500])
    mplt.legend()
    mplt.show()"""

    return p, sum(nb_deces) #renvoie le polynome qui approxime et la somme des deces totaux

def proba_age_sachant_deces(poly_deces, age):
    """age doit être entre 0 et 99"""
    inte = integrale(poly_deces, 0,100)
    proba = integrale(poly_deces, age, age+1)
    return proba/inte


def proba_deces(age, poly_age, poly_deces, nb_tot):
    """age est un entier entre 0 et 110
    p est le polynome d'interpolation renvoyé par lecture_inf
    nb_tot est le sum(nb_inf) de la fonction précédente
    poly_age est le polynome qui donne la probabilité d'être de l'âge âge"""
    if age <= 20 :
        return 0
    tirage = random.random()
    p_age = poly_age(age)
    pdeces = nb_tot/8e6 #probabilité de déceder en géneral, 8e6 = population Quebec
    p_sach = proba_age_sachant_deces(poly_deces, age)
    if p_age != 0 :
        probi =  p_sach * pdeces/p_age  #formule des probabilités conditionelles : p_sach donne la probabilité de age sachant le deces, pdeces est la probabilité de décéder et p_age la probabilité de l'âge, on obtient la probabilité du déces sachant l'âge
        return probi/ t_inf_moy #renvoie la probabilité de déceder sachant l'âge, qu'on considère répartie sur le           temps d'infection
    else :
        return 0



##Caractéristique de chaque personnes

parametre = ["age", "proba_inf", "proba_deces", "dure_inf","dure_immun", "delai"]

p_croisee = 50

def creation_pop(pop):
    """crée un graphe ( en liste d'adjacence de la population et une liste contenant
    chacune des caracteristiques de chaque personne"""

    def creation_personne(age):
        """crée une liste qui contient les caracteristiques d'une personne
        """
        etatp = []
        for i in range(len(parametre)):
            etatp.append(fonction_associé[i](age))
        return etatp
    #Création des graphes de contacts de la population
    gWS =  arrete_to_adj(nx.connected_watts_strogatz_graph(pop, 6, 0.3), pop)
    gBA = arrete_to_adj(nx.barabasi_albert_graph(pop, p_croisee), pop)
    popu  = []
    date_inf = [(-31) for i in range(pop)]


    #determination des polynomes de regression
    p_age = lecture_age()
    p_inf,inf_tot = lecture_inf()
    p_deces, d_tot = lecture_deces()

    #Crétion des fonctions de probabilité
    liste_proba_age = echelle_proba(p_age)#probabilité

    #Normalisation des fonctions
    f4 = lambda x : 30

    I4 = integrale_pas_poly(f4, 0,110)


    phi4 = lambda x : f4(x) / I4

    #Liste des fonctions déterminant les caractéristiques de chaque personne
    fonction_associé = [lambda x : x, lambda age : proba_inf(age, p_age,p_inf, inf_tot), lambda age : proba_deces(age, p_age,p_deces, d_tot) , lambda age : dure_infection(age), lambda x : 30, lambda x : 7]

    #création de la population
    moy = 0
    for i in range(pop):
        print(i)
        age = age_f(p_age, liste_proba_age)
        perso = creation_personne(age) #on associe un âge aléatoire à chaque personne
        popu.append(perso)
        moy += perso[2]

    return gBA, gWS, popu, date_inf


def age_popu(pop):
    """renvoie un graphe de la repartition de l'age dans la population pop"""
    n = len(pop)
    age = [0 for i in range(110)]
    for j in range(n):
        age_p = pop[j][0]
        age[age_p] += 1/n
    mplt.bar([i for i in range(110)],age, label = ' repartition age de la population crée')
    mplt.legend()
    lecture_age()
    mplt.show()

def dure_inf_moy(pop):
    moy = 0
    n = len(pop)
    compte_age = [0 for i in range(110)]
    dure_age = [0 for i in range(110)]
    for i in range(n):
        age = pop[i][0]
        dure_age[age] += pop[i][3]
        compte_age[age] += 1
    for j in range(110):
        if compte_age[j] != 0:
            dure_age[j] = dure_age[j]/compte_age[j]
    mplt.plot([i for i in range(0,110)], dure_age)
    mplt.show()

def dure_inf_mo_age(pop, age):
    nombre = [0 for i in range(100)]
    n = len(pop)
    for i in range(n):
        age_i = pop[i][0]
        if age == age_i :
            nombre[int(pop[i][3])] += 1
    l = """repartition de la duree d'infection de la tranche """ + str(age)
    mplt.plot([i for i in range(100)], nombre, label = l)
    mplt.legend()
    mplt.show()
    return sum(nombre)




##Processus d'infection


dure_moy_inf = 30

def infection_voisin(voisin, etat, l_inf, jour, date_inf, coef_ajustement):
    """infecte la liste de voisins passée en argument, etat est la liste de la population et de leurs caracteristiques"""
    for i in range(len(voisin)):
        """on parcours les personne en contactes avec etat"""
        p = etat[voisin[i]][1] / coef_ajustement #on recupere la probabilité que le voisin soit infecté ajustée par un coefficient pouvant représenter un confinement
        tirage = random.random()
        #valeur des probabilités pour l'individu voisin[i] :
        jour_inf = date_inf[voisin[i]]
        dure_immu = etat[voisin[i]][3]
        #il faut que le voisin ne soit pas déja immunisé suite à une précédente infection et
        #que le voisin ne soit pas infecté actuellement
        if tirage < p and jour_inf + dure_immu < jour and l_inf[voisin[i]] == 0 :
            l_inf[voisin[i]] = 1
            date_inf[voisin[i]] = jour

def infection( gBA, gWS, etat, l_inf, l_inf_suiv, jour, dure_inf, coef_ajus):
    """une journée d'infection, etat est la liste de la population, dure_inf est la liste qui contient les jours d'infection de chaque individu, coef_ajus permet de modifier la probabilité d'infection pour tout le monde pour un éventuel confinement"""
    n = len(etat)
    c = 0 #compteur nouveaux cas
    for i in range(n):
        if l_inf[i] == 1  and etat[i][5] + dure_inf[i] < jour :
                c += 1
                infection_voisin(gBA[i], etat, l_inf_suiv, jour, dure_inf, coef_ajus)
                infection_voisin(gWS[i], etat, l_inf_suiv, jour, dure_inf, coef_ajus)
    return l_inf_suiv, c



## Processus de guerison


def guerison( etat, l_inf, l_inf_suiv, date_inf, jour):
    """etat est la liste de la population"""
    n = len(l_inf)
    cgueri = 0
    for i in range(n):
        #si la personne est dans un etat infectée
        if l_inf[i] == 1 :
            dure_inf = etat[i][3]
            moment_inf = date_inf[i]
            if  jour >= moment_inf + dure_inf:
                #on ne peut guérir que si la durée d'infection minimum est dépassée
                l_inf_suiv[i] = 0
                date_inf[i] = (jour)
                cgueri += 1
    return l_inf_suiv, cgueri


##Processus de mort


def mort( etat, l_inf, l_inf_suiv, date_inf):
    """etat est la liste de la population"""
    n = len(etat)
    compteur = 0
    for i in range(n):
        #si la personne est dans un etat infectée
        if l_inf[i] == 1 :
            proba_mort = etat[i][2]
            tirage = random.random()
            if tirage < proba_mort:  #certaine proba de mourir pendant l'infection
                l_inf_suiv[i] = 2
                date_inf[i] = (-1)
                compteur += 1

    return l_inf_suiv, compteur

## Journée


def journee(gBA,gWS, etat,l_inf, jour, date_inf, coef_ajus):
    """etat est la liste de la population"""
    l_inf_suivante = l_inf.copy()
    l_inf_suivanten, cinf = infection(gBA,gWS, etat,l_inf, l_inf_suivante, jour, date_inf, coef_ajus)
    l_inf_suivante, cmort = mort( etat, l_inf, l_inf_suivante, date_inf)
    l_inf_suivante, cgueri = guerison(etat ,l_inf, l_inf_suivante, date_inf, jour)
    return l_inf_suivante, cgueri, cmort, cinf


##Epidémie

def compte(l_inf):
    """compte le nombre de personnes infectées ou saines dans la liste des infectés"""
    cinf = 0
    cmort = 0
    csains = 0
    for etat in l_inf:
        if etat == 1:
            cinf += 1
        elif etat == 0:
            csains += 1
        elif etat == 2:
            cmort += 1
    return csains, cinf, cmort


def infecte_ini(l, nb_inf, date_inf):
    """infecte un nombre de personne n_inf aléatoirement dans la liste"""
    echantillon = [i for i in range(len(l))]
    k = random.sample(echantillon, nb_inf)
    for i in k:
        l[i] = 1
        date_inf[i] = 0


def epidemieIC(pop, dure):
    """Déroule une épidémie sur la durée"""
    gBA, gWS, etat, date_inf = creation_pop(pop)
    print("ok")
    l_inf = [0 for i in range(pop)]
    sains = []
    mort = []
    mort_totaux = []
    gueri = []
    inf = []
    inf_jour = []
    jour = []

    coef_aju = 1
    infecte_ini(l_inf, 10, date_inf)
    for i in range(dure):
        if i == 100:
            coef_aju = 2
        if i == 200:
            coef_aju = 0.8
        if i == 300:
            coef_aju = 1


        print(i)
        l_inf, cgueri, cmort, cinf_j = journee(gBA,gWS, etat,l_inf, i, date_inf, coef_aju)
        csains, cinf, cmort_tot = compte(l_inf)
        sains.append(csains)
        mort.append(cmort)
        mort_totaux.append(cmort_tot)
        inf_jour.append(cinf_j)
        gueri.append(cgueri)
        inf.append(cinf)
        jour.append(i)
        if cinf == 0:

            return sains,mort, mort_totaux, gueri, inf, inf_jour, jour

    return sains,mort, mort_totaux, gueri, inf, inf_jour, jour



def grapheIC(pop, dure):
    """crée le graphe de l'épidémie au cours du temps"""

    sains,mort, mort_totaux, gueri, inf, cinf_j, jour = epidemieIC(pop, dure)
    l1 = 'infectés'
    l2 = 'sains'
    l3 = 'mort totaux'
    l4 = 'mort par jour'
    l5 = 'guéris par jour'
    mplt.plot(jour, sains, c = "g", label = l2)
    mplt.plot(jour, mort_totaux, c= 'm', label = l3)
    mplt.plot(jour, inf, c = "r", label = l1)
    mplt.legend()
    mplt.show()
    mplt.plot(jour[7:], lissage(gueri), c = "b", label = l5)
    mplt.plot(jour[7:], lissage(mort), c = "k", label = l4)
    mplt.plot(jour[7:], lissage(cinf_j), c = 'r', label = 'nouveaux infectés')
    mplt.legend()
    mplt.show()


def epidemie_moyenneIC():
    moy = [0 for i in range(400)]
    moy_par_j = [0 for i in range(400)]
    jours = [i for i in range(400)]
    for i in range(25):
            print('SIMULATION', i)
            sains,mort, mort_totaux, gueri, inf, inf_jour, jour = epidemieIC(1000, 400)
            for j in range(len(inf)):
                moy[j] += inf[j]/50
                moy_par_j[j] += inf_jour[j]/50

    mplt.plot(jours, moy)
    mplt.show()
    mplt.plot(jours[7:], lissage(moy_par_j), c = 'g')
    mplt.show()


def lire_fich():
    fichier = open("C:\\Users\\jluca\\Documents\\TIPE\\Programmes_et_fichier\\national-activity-indicators.csv")
    for i in range(7):
        fichier.readline()
    y = []
    x = [i for i in range(38)]
    i = 0
    for ligne in fichier:
        valeur = ligne.split(',')[1]
        if valeur == '':
            valeur = 0
        y.append(valeur)

    n = len(x)
    """yy = []
    for i in range( n-7):
        c = 0
        for j in range(7):
            c += y[i + j]
        x.append(k)
        k += 1
        yy.append(c/7)"""
    fichier.close()
    mplt.plot(x,y, label = 'pourcentage de cas positifs')
    mplt.show()
    return yy



### Tracer de graphique

"""mplt.show(graphique_donne(graphique_donnees_covid_gouv))"""

"""graphique_SIR(65e6, 150,170)"""


"""mplt.show(graphique_SEIR(0.29,lambda x : I_SEIR(x,1,0), 400)"""


"""graphique_STO_prem_id(400,10)"""

"""graphique_parcours_Markov()"""

"""graphique_resaux_basique()"""

"""graphique_reseaux_cluster()"""








