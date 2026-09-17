# Fondamenti matematici dei Diffusion Models

Un modello generativo non deve ricostruire un singolo esempio, ma approssimare una distribuzione. Se $x_0\in\mathbb{R}^d$ è un dato osservato, si assume

$$
x_0\sim p_{\mathrm{data}}(x).
$$

L'obiettivo è costruire un modello $p_\theta$ dal quale ottenere nuovi campioni plausibili. I diffusion model rendono il problema trattabile introducendo una famiglia di distribuzioni $p_t$, con $p_0=p_{\mathrm{data}}$ e $p_T$ vicino a un prior semplice.

## Processo forward discreto

Nel DDPM classico il processo forward è una catena di Markov:

$$
q(x_{1:T}\mid x_0)
=
\prod_{t=1}^{T}q(x_t\mid x_{t-1}).
$$

Ogni transizione aggiunge una piccola quantità di rumore gaussiano:

$$
q(x_t\mid x_{t-1})
=
\mathcal{N}
\left(
x_t;
\sqrt{1-\beta_t}\,x_{t-1},
\beta_t I
\right),
$$

dove $\beta_t\in(0,1)$ controlla la varianza introdotta al passo $t$. Definendo

$$
\alpha_t=1-\beta_t,
\qquad
\bar\alpha_t=\prod_{s=1}^{t}\alpha_s,
$$

la marginale condizionata al dato iniziale possiede una forma chiusa:

$$
q(x_t\mid x_0)
=
\mathcal{N}
\left(
x_t;
\sqrt{\bar\alpha_t}\,x_0,
(1-\bar\alpha_t)I
\right).
$$

Di conseguenza si può campionare direttamente

$$
x_t
=
\sqrt{\bar\alpha_t}\,x_0
+
\sqrt{1-\bar\alpha_t}\,\epsilon,
\qquad
\epsilon\sim\mathcal{N}(0,I).
$$

Questa identità evita di simulare tutti i passaggi precedenti durante il training. La proprietà di Markov definisce la catena, mentre la chiusura delle gaussiane rende efficiente il campionamento delle sue marginali.

## Segnale, rumore e SNR

È utile introdurre

$$
\alpha(t)=\sqrt{\bar\alpha_t},
\qquad
\sigma(t)=\sqrt{1-\bar\alpha_t},
$$

così che

$$
x_t=\alpha(t)x_0+\sigma(t)\epsilon.
$$

Il **signal-to-noise ratio** è

$$
\operatorname{SNR}(t)
=
\frac{\alpha(t)^2}{\sigma(t)^2}.
$$

Per $t$ piccolo domina il segnale e il task consiste nel correggere dettagli fini. Per $t$ grande domina il rumore e il modello deve ricostruire la struttura globale. Lo schedule non distribuisce quindi soltanto la quantità di rumore: distribuisce differenti problemi di predizione lungo il tempo.

Il **log-SNR**

$$
\lambda(t)=\log\frac{\alpha(t)^2}{\sigma(t)^2}
$$

è spesso una coordinata più conveniente. Permette di confrontare schedule discreti e continui senza dipendere direttamente dalla particolare parametrizzazione del tempo.

## Processo inverso

La generazione richiede di approssimare le transizioni inverse. Si introduce

$$
p_\theta(x_{0:T})
=
p(x_T)
\prod_{t=1}^{T}p_\theta(x_{t-1}\mid x_t),
$$

con $p(x_T)=\mathcal{N}(0,I)$. Nel DDPM, ogni transizione inversa viene modellata come gaussiana:

$$
p_\theta(x_{t-1}\mid x_t)
=
\mathcal{N}
\left(
x_{t-1};
\mu_\theta(x_t,t),
\Sigma_\theta(x_t,t)
\right).
$$

Il modello non deve memorizzare una traiettoria inversa per ogni esempio. Deve apprendere, per ogni $x_t$ e livello di rumore, l'informazione locale che rende plausibile il passo verso $p_{t-1}$.

## Score di una distribuzione

Per una densità differenziabile $p(x)$, lo **score** è

$$
s(x)=\nabla_x\log p(x).
$$

Lo score non misura la qualità di un campione. È un vettore che indica la direzione locale di massima crescita della log-densità. Per una gaussiana isotropa

$$
p(x)=\mathcal{N}(x;\mu,\sigma^2I),
$$

si ottiene

$$
\nabla_x\log p(x)
=
-\frac{x-\mu}{\sigma^2}.
$$

Il vettore punta verso la media e la sua intensità cresce con la distanza, scalata dalla varianza. Nei diffusion model interessa lo score delle marginali rumorose:

$$
s_t(x_t)=\nabla_{x_t}\log p_t(x_t).
$$

Poiché $p_t$ cambia con il livello di rumore, anche il campo deve dipendere da $t$.

## Collegamento tra noise prediction e score

Condizionatamente a $x_0$, la distribuzione di $x_t$ è gaussiana. Il suo score è

$$
\nabla_{x_t}\log q(x_t\mid x_0)
=
-\frac{x_t-\alpha(t)x_0}{\sigma(t)^2}.
$$

Usando $x_t=\alpha(t)x_0+\sigma(t)\epsilon$ segue

$$
\nabla_{x_t}\log q(x_t\mid x_0)
=
-\frac{\epsilon}{\sigma(t)}.
$$

Predire $\epsilon$ equivale quindi a predire lo score condizionale a meno di un fattore noto. Durante la generazione serve però lo score marginale $\nabla_{x_t}\log p_t(x_t)$, perché $x_0$ è ignoto. Il risultato alla base del **denoising score matching** è che la regressione media sui campioni perturbati produce proprio l'estimatore marginale necessario:

$$
s_\theta(x_t,t)
\approx
\nabla_{x_t}\log p_t(x_t).
$$

Il dato pulito viene dunque usato nel training per costruire un target accessibile, ma non è richiesto in inferenza.

## Parametrizzazioni equivalenti

Dato $x_t=\alpha x_0+\sigma\epsilon$, il network può predire differenti quantità.

La **noise prediction** restituisce $\hat\epsilon_\theta$. Una stima del dato pulito si ricava come

$$
\hat x_{0,\theta}
=
\frac{x_t-\sigma\hat\epsilon_\theta}{\alpha}.
$$

La **data prediction** produce direttamente $\hat x_{0,\theta}$. Il rumore corrispondente è

$$
\hat\epsilon_\theta
=
\frac{x_t-\alpha\hat x_{0,\theta}}{\sigma}.
$$

Una terza scelta è la **velocity prediction**

$$
v=\alpha\epsilon-\sigma x_0.
$$

Quando $\alpha^2+\sigma^2=1$, le trasformazioni sono

$$
x_0=\alpha x_t-\sigma v,
\qquad
\epsilon=\sigma x_t+\alpha v.
$$

Le parametrizzazioni contengono idealmente la stessa informazione, ma una loss MSE uniforme sul target induce pesi effettivi differenti sui livelli di rumore. Per questo la scelta del target influenza l'ottimizzazione anche quando le conversioni algebriche sono esatte.

## Dal tempo discreto al tempo continuo

La sequenza $p_0,p_1,\ldots,p_T$ può essere sostituita da una famiglia continua $p_t$, con $t\in[0,1]$. Un processo stocastico può essere scritto come

$$
d x
=
f(x,t)\,dt
+
g(t)\,dW_t,
$$

dove $f$ è il drift, $g$ il coefficiente di diffusione e $W_t$ un processo di Wiener. Nel limite di passi piccoli, differenti schedule discreti corrispondono a particolari scelte di $f$ e $g$.

Questa formulazione rivela due possibili dinamiche inverse. La **reverse-time SDE** conserva una componente casuale e usa lo score; la **probability flow ODE** è deterministica ma possiede le stesse marginali temporali. La derivazione completa viene sviluppata nel capitolo sui modelli score-based.

## Dallo score alla velocità

Una famiglia $p_t$ può essere descritta anche attraverso un campo di velocità $u_t(x)$ che soddisfa l'equazione di continuità:

$$
\partial_t p_t(x)
+
\nabla_x\cdot\bigl(p_t(x)u_t(x)\bigr)
=0.
$$

Lo score è il gradiente della log-densità; la velocità specifica invece come si muovono le particelle nel tempo. Nei probability path gaussiani i due oggetti possono essere convertiti l'uno nell'altro mediante coefficienti noti, ma rappresentano domande concettualmente diverse.

Questa distinzione prepara il passaggio al Flow Matching: anziché apprendere lo score e derivare successivamente una dinamica di sampling, si può addestrare direttamente il campo vettoriale di una ODE che trasporta il prior verso i dati.

## Quadro concettuale

Le relazioni fondamentali possono essere riassunte così:

$$
\text{processo di rumore}
\Longrightarrow
\{p_t\}_{t\in[0,1]}
\Longrightarrow
\begin{cases}
\nabla_x\log p_t(x) & \text{score},\\
u_t(x) & \text{velocità}.
\end{cases}
$$

Il DDPM discretizza il processo e apprende una transizione inversa; gli score-based model descrivono la dinamica mediante SDE e ODE; il Flow Matching costruisce un probability path e ne regredisce la velocità. Comprendere quali quantità siano fissate e quali apprese evita di trattare queste formulazioni come algoritmi privi di una struttura comune.
