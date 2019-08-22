
# <center> <h1> Echantillonnage </h1> </center>

<img src="./images/sample.png" width="300">


```python
get_ipython().magic(u'matplotlib inline')
%run -i ./utils/credentials.py
%run -i ./utils/imports.py
%run -i ./utils/plots.py
%run -i ./utils/stats.py
```


        <script type="text/javascript">
        window.PlotlyConfig = {MathJaxConfig: 'local'};
        if (window.MathJax) {MathJax.Hub.Config({SVG: {font: "STIX-Web"}});}
        if (typeof require !== 'undefined') {
        require.undef("plotly");
        requirejs.config({
            paths: {
                'plotly': ['https://cdn.plot.ly/plotly-latest.min']
            }
        });
        require(['plotly'], function(Plotly) {
            window._Plotly = Plotly;
        });
        }
        </script>
        


# Dans ce qui suit on va voir : 


* ce qu'est un échantillon et dans quels cas échantillonner

* des méthodes pour choisir la taille d'un échantillon

* des méthodes pour récupérer des données auprès de tierces personnes

* des cas d'utilisation d'échantillons bootsrappés

* du code pour faire tout ça

## Définitions
L'échantillonnage est un ensemble de techniques permettant d'extraire des individus à partir d'un ensemble. Cette extraction est un __échantillon__.

L'ensemble des individus est appelé la __population totale__. Il correspond à l'ensemble des individus réels ou potentiels.

L'échantillon tiré doit avoir une caractéristique essentielle : __la représentativité__.
Cela signifie que les résultats calculés à partir de l'échantillon doivent être proches de ceux qui seraient obtenus à partir de la population totale.

Le principe général est de procéder de manière la plus __aléatoire__ possible. Ceci vient du fait que l'aléatoire pur permet d'éviter les biais. C'est donc une condition essentielle pour que l'échantillon soit représentatif de l'ensemble des données.

# Les use-cases

## A/ Impossible d'utiliser toutes les données

Il est parfois impossible ou trop coûteux d'utiliser toutes les données.

__Données difficile à rassembler__

Les données ne sont pas toujours toutes disponibles : 
* détruites régulièrement ;
* inexistantes ;
* complexes et longues à récupérer (stockage décentralisé, formats hétérogènes)

__Traitements trop coûteux__

Il est parfois très coûteux de traiter les données : 
* en temps ;
* en argent.

Dans tous ces cas il peut être utile de procéder à un échantillonnage pour réduire le coût de récupération et de traitement des données.

<img src="./images/warning.png" width="100">
Lorsque l'on peut travailler sur toutes les données c'est toujours mieux !

## B/ Améliorer ou estimer des performances - bootstrap

La technique du boostrap peut être utilisée dans certains cas pour :
* améliorer les performances d'un algorithme de machine learning : bagging (boostrap aggregating) ;
* estimer des intervalles de confiance et plus généralement la stabilité d'un modèle.

# La base de sondage

Pour commencer, il faut une base de sondage, c'est-à-dire un minimum d'informations sur l'ensemble de la population, afin de tirer les individus échantillonnés.

__NB__ : La base de sondage ne doit pas être partielle sinon on risque d'avoir un biais de sélection des individus.

# La taille de l'échantillon

D'un point de vue statistique, plus on a d'indivdus, mieux c'est. 

Mais l'échantillonnage a un coût. 

Il va donc falloire arbitre entre coût et précision des résultats. Pour cela on va estimer la précision que l'on peut espérer pour les résultats qui seront calculés grâce à l'échantillon.

##### Exemple : calcul d'un taux pour un tirage aléatoire simple

On souhaite par exemple calculer le taux de clics pour les visiteurs de notre site. Soient $X = (X_1,...,X_n)$ les variables aléatoires i.i.d représentant les $n$ visiteurs de notre échantillon. On considère que tous les visiteurs ont une probabilité de cliquer valant $p$. 


On a donc que :
* la probabilité qu'un visteur $i$ clique, $P(X_i=1)$, est $p$ ;
* et la probabilité qu'il ne clique pas, $P(X_i=0)$, est $(1-p)$. 

Autrement dit on $$\forall i, X_i \sim \mathcal{B}(p)$$

Le taux de clic de notre échantillon est la moyenne des $X_i$ notée $\bar{X}$

Lorsque l'on calcule une moyenne empirique sur un échantillon, la valeur s'écarte de la _vraie valeur_ (sur l'ensemble de la population ou théorique). Ce que l'on cherche à prévoir ici c'est de combien on risque de se tromper pour une taille d'échantillon donnée.
Pour cela on utilise (encore) le théorème central limite :

$$ \frac{\bar{X} - \mu}{\sqrt{\frac{\sigma^2}{n}}} \rightarrow \mathcal{N}\Big(0, 1\big)$$

Qui est équivalent à :

$$ \bar{X} - \mu \rightarrow \mathcal{N}\Big(0, \sqrt{\frac{\sigma^2}{n}} \Big)$$

Dans notre cas, comme on a $\forall i, X_i \sim \mathcal{B}(p)$, on sait que la variance $\sigma^2$ vaut $p(1-p)$.

Pour chaque valeur de $n$ on peut donc calculer un intervalle de confiance, à $95$% par exemple, de l'erreur que l'on fera. A 95%, l'intervalle de confiance d'une variable aléatoire suivant une loi $\mathcal{N}\big(0, 1\big)$ est $[-1,96 ; 1,96]$. 

Dans 95% des cas on aura donc : 

$$ \frac{|\bar{X} - p|}{\sqrt{\frac{p(1-p)}{n}}} \leq 1,96$$
$$ \iff n \geq 1.96^2 \times \frac{p(1-p)}{(\bar{X} - p)^2}$$

Si on veut avoir une erreur $|\bar{X}-p|$ plus petite qu'une valeur $e$ dans 95% des cas, il faut avoir une taille d'échantillon $n$ suffisamment grande.

$$ |\bar{X} - p|<e$$
$$ \Rightarrow n \geq 1.96^2 \times \frac{p(1-p)}{(e)^2}$$


```python
def n_size_proportion(p,e,i=0.95):

    # mu : hypothèse sur la valeur réelle
    # e : erreur moyenne(X) - mu // l'écart entre la moyenne et mu sera de +/- e 
    # i : intervalle de confiance
    
    ii = 1 - (1 - i)/2
    
    ppf = norm.ppf(ii) # pour i = 95%, la valeur est de 1,96
    
    n = (ppf**2)*(p*(1-p))/((e)**2)
    
    return n
```


```python
n_size_proportion(0.5,0.05,i=0.95)
```




    384.14588206941244




```python
columns = []
for pp in range(1,10):
    columns.append("p:{}%".format(pp*10))
index = []
for ee in range(1,10):
    index.append("error:{}%".format(ee))

```


```python
import pandas as pd
matrix = []
for ee in range(1,10):
    line = []
    for pp in range(1,10):
         line.append(n_size_proportion(pp/10, ee/100, 0.95))
    matrix.append(line)
```


```python
table = pd.DataFrame(matrix,columns=columns, index=index)
```


```python
table.to_csv('./table_echantillon')
```


```python
table
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>p:10%</th>
      <th>p:20%</th>
      <th>p:30%</th>
      <th>p:40%</th>
      <th>p:50%</th>
      <th>p:60%</th>
      <th>p:70%</th>
      <th>p:80%</th>
      <th>p:90%</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>error:1%</th>
      <td>3457.312939</td>
      <td>6146.334113</td>
      <td>8067.063523</td>
      <td>9219.501170</td>
      <td>9603.647052</td>
      <td>9219.501170</td>
      <td>8067.063523</td>
      <td>6146.334113</td>
      <td>3457.312939</td>
    </tr>
    <tr>
      <th>error:2%</th>
      <td>864.328235</td>
      <td>1536.583528</td>
      <td>2016.765881</td>
      <td>2304.875292</td>
      <td>2400.911763</td>
      <td>2304.875292</td>
      <td>2016.765881</td>
      <td>1536.583528</td>
      <td>864.328235</td>
    </tr>
    <tr>
      <th>error:3%</th>
      <td>384.145882</td>
      <td>682.926013</td>
      <td>896.340391</td>
      <td>1024.389019</td>
      <td>1067.071895</td>
      <td>1024.389019</td>
      <td>896.340391</td>
      <td>682.926013</td>
      <td>384.145882</td>
    </tr>
    <tr>
      <th>error:4%</th>
      <td>216.082059</td>
      <td>384.145882</td>
      <td>504.191470</td>
      <td>576.218823</td>
      <td>600.227941</td>
      <td>576.218823</td>
      <td>504.191470</td>
      <td>384.145882</td>
      <td>216.082059</td>
    </tr>
    <tr>
      <th>error:5%</th>
      <td>138.292518</td>
      <td>245.853365</td>
      <td>322.682541</td>
      <td>368.780047</td>
      <td>384.145882</td>
      <td>368.780047</td>
      <td>322.682541</td>
      <td>245.853365</td>
      <td>138.292518</td>
    </tr>
    <tr>
      <th>error:6%</th>
      <td>96.036471</td>
      <td>170.731503</td>
      <td>224.085098</td>
      <td>256.097255</td>
      <td>266.767974</td>
      <td>256.097255</td>
      <td>224.085098</td>
      <td>170.731503</td>
      <td>96.036471</td>
    </tr>
    <tr>
      <th>error:7%</th>
      <td>70.557407</td>
      <td>125.435390</td>
      <td>164.633949</td>
      <td>188.153085</td>
      <td>195.992797</td>
      <td>188.153085</td>
      <td>164.633949</td>
      <td>125.435390</td>
      <td>70.557407</td>
    </tr>
    <tr>
      <th>error:8%</th>
      <td>54.020515</td>
      <td>96.036471</td>
      <td>126.047868</td>
      <td>144.054706</td>
      <td>150.056985</td>
      <td>144.054706</td>
      <td>126.047868</td>
      <td>96.036471</td>
      <td>54.020515</td>
    </tr>
    <tr>
      <th>error:9%</th>
      <td>42.682876</td>
      <td>75.880668</td>
      <td>99.593377</td>
      <td>113.821002</td>
      <td>118.563544</td>
      <td>113.821002</td>
      <td>99.593377</td>
      <td>75.880668</td>
      <td>42.682876</td>
    </tr>
  </tbody>
</table>
</div>



# Les techniques d'échantillonnage

Le tirage aléatoire simple est une technique qui doit toujours fonctionner en principe. Mais s'il n'existe pas de bon arbitrage entre taille et coût on peut envisager d'autres solutions.

__Le tirage aléatiore simple__

<img src="./images/sample.png" width="400">

Cette méthode est la plus simple et permet une bonne représentation non biaisée de l'ensemble des données, à condition d'avoir un échantillon de taille suffisante.

Nous avons vu plus haut comment calculer la taille de l'échantillon dans ce cas.

__L'échantillonnage par strate__

<img src="./images/sample_strates.png" width="400">

_Principe_

Le principe d'un tirage par strate est de réaliser un tirage aléatoire au sein de sous-groupe de population plus homogène afin d'améliorer la précision des résultats.

_Méthode_

* __On divise la population en strates__ : Les strates doivent constituer une partition des données, c'est-à-dire que chaque individu appartient à une et une seule strate.

* __On calcule le poids de chaque strate__ : pour chaque strate on calcule le rapport du nombre d'individus dans la strate sur la taille de la population totale.

* __On réalise un tirage aléatoire par strate__ : le tirage aléatoire au sein de chaque strat est réalisé en proportion de la strat dans la population totale.

__L'échantillonnage par strate__

<img src="./images/sample_strates.png" width="400">

_Avantages_

Cette méthode est meilleure qu'un tirage aléatoire simple (au sens de la variance de la variable observée) si la variance intra-strate est faible.

_Limites_

Cette méthode nécessite que les informations de la base de sondage soient suffisantes pour regrouper les individus selon des caractéristiques __pertinentes__, c'est-à-dire permettant de créer des groupes homogènes. Cela ne sert _a priori_ à rien de regrouper les individus en fonction de la valeur du dernier chiffre de leur numéro de téléphone.

_Exemple_

Si l'on souhaite calculer le revenu moyen des franciliens et si on dispose d'une base de sondage avec les CSP de tous les habitants d'Ile de France, on peut réaliser un échantillonnage stratifié sur la csp.

__L'échantillonnage par grappe__

<img src="./images/sample_cluster.png" width="400">

_Principe_

L'échantillonnage par grappe consiste à réaliser un tirage aléatoire de groupes d'individus et de retenir dans l'échantillon tous les individus des groupes échantillonnés.

_Exemple_

Par exemple pour obtenir des information sur des logements sociaux il est efficace de tirer de manière aléatoire des bailleurs sociaux (il est couteux de leur demander de données) et de récupérer les données pour tous leurs logements (extraire pour 1 ou 1000 logements revient au même).

_Avantage/limtes_

Cette méthode est moins bonne mais a l'avantage d'être parfois très économique.

_Remarque_

Le principe est de faire porter le tirage aléatoire sur l'étape la plus coûteuse.

__Les quotas__

<img src="./images/sample_quota.png" width="400">

On __choisit__ des individus dans une liste en fonction de leurs caractéristiques pour coller aux caractéristiques principales de l'ensemble des données (population générale).

Cette méthode introduit plusieurs biais :
* les mêmes personnes sont souvent interrogées (il est moins cher d'interroger des meilleurs répondants) ;
* souvent peu d'individus représentent toutes leur catégorie.

La méthode des quotas présente également l'incovénient d'empêcher de calculer sérieuseument les erreurs que l'on va avoir sur les résultats.

__Techniques combinées__

Il est possible de combiner des méthodes d'échantillonnage, par exemple en réalisant un tirage stratifié comme première étape d'un tirage par grappe.

### Cas pratique : comparaison d'un tirage aléatoire simple et d'un tirage stratifié


```python
data_path = "./data/PUBG_train_sample.csv"
data_PUBG = pd.read_csv(data_path)
```


```python
q = 0.6
break_point = np.quantile(data_PUBG["killPoints"], q=q)
data_PUBG["is_top_killer"] = data_PUBG["killPoints"] > break_point
```


```python
sampling_frame = data_PUBG[["Id","is_top_killer"]]
```


```python
def sample_random(sampling_frame, data= data_PUBG, sample_size=1000):
    # tirage dans la base de sondage
    sample_random = sampling_frame.sample(sample_size)
    # récupération des données dans la population générale
    sample_random = sample_random.merge(data_PUBG, on = "Id", how="left")
    
    return sample_random
```


```python
data_sample_random = sample_random(sampling_frame)
np.mean(data_sample_random["winPoints"])
```




    589.799




```python
def sample_strates(sampling_frame, data= data_PUBG, sample_size = 1000):
    n = len(sampling_frame)

    strate_1_bool = sampling_frame["is_top_killer"]
    strate_2_bool = sampling_frame["is_top_killer"]==False
    
    # weight of each strate
    weight_1 = sum(strate_1_bool)/n
    weight_2 = 1 - weight_1
    
    # size of the sample for each strate
    size_1 = int(sample_size*weight_1)
    size_2 = sample_size - size_1

    # sample on each strat in proportion
    sample_1 = sampling_frame[strate_1_bool].sample(size_1)
    sample_2 = sampling_frame[strate_2_bool].sample(size_2)

    # concatenate samples
    sample_strates = pd.concat([sample_1,sample_2])
    sample_strates = sample_strates.merge(data_PUBG, on = "Id", how="left")
    
    return sample_strates
```


```python
data_sample_strates = sample_strates(sampling_frame)
np.mean(data_sample_strates["winPoints"])
```




    600.502



Comparaison des résultats


```python
np.mean(data_PUBG["winPoints"]) - np.mean(data_sample_random["winPoints"])
```




    12.94380000000001




```python
np.mean(data_PUBG["winPoints"]) - np.mean(data_sample_strates["winPoints"])
```




    2.2408000000000357



# Collecte des données (si besoin)

Si les données doivent être collectées auprès de tiers ou sont le résultat d'un questionnaires quelques problèmes peuvent se poser :
* les non réponses ;
* les formats hétérogènes ;
* les réponses erronées.

Pour minimiser les risques de rencontrer ces difficultés quelques principes peuvent être suivis.

__Sensibilisation des interlocuteurs__

La qualité des données et de l'échantillon dépend souvent d'autres acteurs : personnes enquêtées, celles qui trasmettent les données etc. La sensibilitation de ces acteurs est essentielle. 

Expliquer les pourquoi et comment du projet, les objectifs, les résultats attendus permettra d'abord de montrer à votre interlocuteur que vous le considérer comme un être humain (et pas seulement un moyen). Ensuite, cela permettra de montrer pourquoi les informations demandées sont importantes, quelles sont les retombées positives attendues, etc.

_Plus les autres acteurs sont impliqués plus vous serez susceptibles d'avoir des réponses et des réponses de qualité._

__Préparation d'un questionnaire__

En général c'est une bonne idée de se faire aider par un.e professionnel.le lorsque l'on rédige un question et de le préparer avec soin pour être sûr que les informations demandées sont nécessaires et suffisantes à l'objectif du projet.

_A/ Prévoir une phase d'entretiens qualitatifs_. 

Ces entretiens doivent permettre d'établir des hypothèse que le questionnaire à proprement parler va chercher à vérifier.

_B/ Appliquer quelques principes pour un questionnaire_. 

Les questions doivent :
* être claires (univalentes)
* être courtes
* répondre à but précis
* le moins nombreuses possibles

Il peut être utile de garder des questions ouvertes pour explorer le sujet sans oeillères, mais en petit nombre.

__Formulation de la demande de données__

Faire une demande pertinente nécessite souvent de la "jouer fine". Vous aller demander à vos interlocuteurs de réaliser un travail : 
* qui souvent n'est pas valorisé ;
* qui leur nécessite du temps ;
* qui dépasse leurs compétences ;
* qu'il ne veulent pas faire car ils veulent que votre projet capote.

_Tips_

Pour réaliser une demande de données, il est parfois crucial de comprendre le contexte institutionnel, de connaître les actions possibles en cas de non réponse ou de refus. Il peut être aussi très utile d'avoir des "alliés" qui vont vous mettre au parfum et appuyer votre demande.

_Relances_

Les délais doivent être clairs et fixes et des rappels doivent être réalisés suffisamment tôt pour que l'interlocuteur ait le temps de répondre dans les temps.

La fréquence et l'intensité des relances dépend largement du contexte.

#### Prévoir des remplaçants en cas de non réponse

Pour éviter d'avoir un volume bien moins important à l'arrivée que celui recherché, il est possible de prévoir des remplçants. Si un individu ne répond pas, un autre aux caractéristiques similaires aura déjà été choisi pour le remplacer

# Traitement des données - le boostrapping

__Le principe du boostrap__

<img src="./images/sample_bootstrap.png" width="400">



_Problème_

Dans l'idéal on souhaite tirer un grand nombre d'échantillons pour :
* estimer un intervalle de confiance des résultats (calcul de moyenne, modèle, etc.) de manière empirique ;
* vérifier la stabilité de résultats ;
* améliorer des algorithmes en utilisant la technique du bagging (sous certaines conditions), comme dans le random forest.

Le problème est qu'il est parfois coûteux voire impossible de tirer plusieurs échantillons.



_Solution_

La solution est de tirer des échantillons, dits bootstrappés, à partir de l'échantillon de départ.

Le bootstrap s'effectue en trois étapes : 
* 1/ d'abord on tire, __de manière aléatoire et avec remise__ $m$ échantillons de même taille $n$ que l'échantillon initial ;
* 2/ ensuite on calcule les résultats sur chacun des échantillons bootstrappés ;
* 3/ enfin on réalise le traitement (aggrègation des résultats ou calcul de l'intervalle).

__NB__
Les tirages se font avec remises car sinon on obtient $m$ fois le même échantillon.

### Cas pratique : calcul d'un intervalle de confiance autour d'une moyenne


```python
import numpy as np
mean = 0
std = 1
sample_size = 1000
sample = np.random.normal(mean, std, size=sample_size)
```


```python
def bootstrap_mean(sample, boot_size = 1000):
    n = len(sample)
    mean_boot = []
    for ii in range(boot_size) :
        sample_boot = np.random.choice(sample, n, replace=True)
        mean_boot.append(np.mean(sample_boot))
    return mean_boot
```


```python
boot_mean = bootstrap_mean(sample, boot_size = 5000)
```


```python
np.mean(sample)
```




    -0.0029865712001975525




```python
np.quantile(boot_mean, q=[0.025,0.975])
```




    array([-0.064094  ,  0.05952682])



<center> <h1>Take away</h1> </center>

<img src="./images/coffee.png" width="200">


__Expresso__ : 

* Il faut de l'aléa pour être représentatif
* Plusieurs techniques permettent de trouver un équilibre entre coût de l'échantillonnage et représentativité
* La représentativité : la taille ça compte mais ce n'est pas suffisant
* La taille de l'échantillon total n'est pas nécessaire pour calculer la taille de l'échantillon cible

<center> <h1>Take away</h1> </center>

<img src="./images/coffee.png" width="200">

__Sugar Story__ :

* Biais : sondage Gallup éléction US de 1936 (https://www.lemonde.fr/economie/article/2005/05/27/l-histoire-debute-avec-l-institut-gallup-et-l-election-du-president-roosevelt_654835_3234.html)
* Taille de l'échantillon : métaphore de la soupe

# Get more on my github <img src="./images/github.png" width="100">
https://github.com/JJublanc/statistics_tools


```python
# jupyter nbconvert --to slides sampling.ipynb --post serve 
```
