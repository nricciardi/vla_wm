# Training ed engineering dei modelli diffusivi

La loss di un diffusion model è compatta, ma un'implementazione efficace dipende da molte decisioni che la formula astratta non mostra. **Schedule, parametrizzazione, loss weighting, normalizzazione, precisione, EMA e sampler** interagiscono tra loro. Per questo una configurazione deve essere descritta come un sistema, non come un elenco di hyperparameter indipendenti.

## Pipeline minima di training

Per un modello gaussiano condizionato, un passo di training può essere espresso così:

```python
x0 = normalize(batch["sample"])
condition = encode_condition(batch)
t = sample_timesteps(x0.shape[0])
noise = randn_like(x0)
xt = alpha(t) * x0 + sigma(t) * noise

target = make_target(x0, noise, t)
prediction = model(xt, t, condition)
loss = weighted_mse(prediction, target, t)

optimizer.zero_grad(set_to_none=True)
loss.backward()
clip_grad_norm_(model.parameters(), max_norm)
optimizer.step()
ema.update(model)
```

Ogni funzione nasconde una scelta scientifica. `normalize` definisce la geometria dello spazio; `sample_timesteps` decide quali livelli di rumore ricevono più update; `make_target` distingue $\epsilon$, $x_0$, $v$ e velocity di Flow Matching; `weighted_mse` determina il contributo effettivo di ciascun timestep.

## Normalizzazione dei dati

Il rumore isotropo $\mathcal{N}(0,I)$ presuppone scale comparabili tra dimensioni. Per immagini si usa spesso un intervallo simmetrico, per esempio $[-1,1]$. Nei latent si applica lo scaling previsto dall'autoencoder. Nelle azioni robotiche, traslazioni, rotazioni, giunti e gripper devono essere normalizzati con statistiche robuste e coerenti con l'embodiment.

Se una coordinata ha varianza molto maggiore, domina la MSE e riceve più capacità. Clipping per quantili può limitare outlier, ma deve essere documentato perché modifica la distribuzione target. Le trasformazioni vanno invertite prima di valutare errori fisici o inviare comandi al robot.

## Noise schedule

Uno schedule discreto specifica $\beta_t$ oppure, equivalentemente, $\bar\alpha_t$. Uno schedule continuo specifica $\alpha(t)$ e $\sigma(t)$. La quantità realmente informativa è spesso il log-SNR:

$$
\lambda(t)
=
\log\frac{\alpha(t)^2}{\sigma(t)^2}.
$$

Uno schedule lineare in $\beta_t$ non è lineare in log-SNR. Può quindi dedicare molti timestep a regioni nelle quali il dato è già quasi distrutto. Lo **schedule cosine** rallenta la perdita di informazione nelle fasi iniziali e distribuisce diversamente la difficoltà.

Per latent, immagini ad alta risoluzione e action chunk, lo schedule ottimale può cambiare perché cambia la dimensionalità efficace e la distribuzione delle frequenze. Copiare uno schedule senza conservare preconditioning e parametrizzazione rende il confronto poco significativo.

## Campionamento dei timestep

Campionare $t$ uniformemente non implica pesare uniformemente i problemi. Se il mapping $t\mapsto\lambda(t)$ è non lineare, alcune regioni del log-SNR vengono osservate più spesso. Si può campionare direttamente una distribuzione su $t$, $\sigma$ o log-SNR e correggere eventualmente la loss tramite importance weighting.

Nei rectified flow text-to-image, distribuzioni non uniformi possono enfatizzare scale percettivamente rilevanti. Nei task robotici è necessario verificare che il bias temporale non penalizzi la precisione finale delle azioni, soprattutto vicino all'estremo pulito del path.

## Loss weighting

Una loss generale assume la forma

$$
\mathcal{L}(\theta)
=
\mathbb{E}_{t,x_0,\epsilon}
\left[
w(t)
\left\|y_t-y_\theta(x_t,t,c)\right\|_2^2
\right].
$$

Il peso $w(t)$ può derivare dalla ELBO, da considerazioni sullo score o da un criterio empirico. Anche con $w(t)=1$, cambiare target modifica il peso implicito sull'errore di ricostruzione di $x_0$.

Il **Min-SNR weighting** limita il contributo delle regioni a SNR elevato, nelle quali obiettivi associati a timestep differenti possono entrare in conflitto. Non è una regola universale: soglia, target e schedule devono essere considerati insieme. Il riferimento è [Efficient Diffusion Training via Min-SNR Weighting Strategy](https://arxiv.org/abs/2303.09556).

È utile monitorare la loss per bin di log-SNR. Una sola media può nascondere un modello molto accurato a rumore alto ma debole nella ricostruzione finale, o il comportamento opposto.

## Preconditioning

Quando la scala di $x_t$ cambia molto con $t$, il network riceve input con statistiche diverse e deve produrre target di ampiezza diversa. Il **preconditioning** riscrive il modello usando coefficienti analitici:

$$
D_\theta(x;\sigma)
=
c_{\mathrm{skip}}(\sigma)x
+
c_{\mathrm{out}}(\sigma)
F_\theta(c_{\mathrm{in}}(\sigma)x,c_{\mathrm{noise}}(\sigma)).
$$

I coefficienti controllano scala dell'input, skip connection e output. Una scelta ben condizionata rende l'ottimizzazione più uniforme tra noise level. [Elucidating the Design Space of Diffusion-Based Generative Models](https://arxiv.org/abs/2206.00364) separa esplicitamente preconditioning, schedule, loss e sampler, mostrando perché non vadano trattati come un unico algoritmo monolitico.

## Exponential Moving Average

Durante il training, i parametri oscillano a causa del rumore stocastico dei minibatch. L'**Exponential Moving Average (EMA)** mantiene una copia

$$
\theta_{\mathrm{EMA}}
\leftarrow
\gamma\theta_{\mathrm{EMA}}
+
(1-\gamma)\theta,
$$

con decay $\gamma$ vicino a uno. Il checkpoint EMA produce spesso campioni più stabili del modello istantaneo.

Il decay deve essere interpretato rispetto al numero di update e alla batch size effettiva. In distributed training, modificare il numero di worker cambia il numero di esempi osservati tra due aggiornamenti EMA. Occorre salvare sia pesi raw sia EMA, oltre allo stato dell'optimizer, per poter riprendere correttamente il training.

## Precisione numerica

La mixed precision riduce memoria e aumenta throughput, ma non tutte le operazioni sono ugualmente sicure. **bfloat16** conserva l'intervallo esponenziale di float32 ed è spesso più robusto di float16; quest'ultimo può richiedere loss scaling dinamico.

Riduzioni della loss, statistiche di normalizzazione, coefficienti dello schedule e aggiornamenti dell'optimizer possono essere mantenuti in float32. Agli estremi temporali, rapporti che coinvolgono $\alpha(t)$ o $\sigma(t)$ possono produrre divisioni instabili. Clamping motivato e limiti temporali espliciti sono preferibili a correzioni silenziose di NaN.

Il sampling è a sua volta sensibile alla precisione: integrare molti piccoli aggiornamenti in bassa precisione può accumulare errore. Per action generation, un errore apparentemente piccolo nello spazio normalizzato può diventare rilevante dopo la denormalizzazione.

## Stabilità dell'ottimizzazione

AdamW è una scelta comune, con warmup iniziale e decay cosine o costante. Il learning rate deve essere interpretato insieme a batch size, gradient accumulation e numero di token o pixel per batch. Il **gradient clipping** limita update eccezionali ma non corregge una loss mal scalata.

Activation checkpointing riduce memoria ricalcolando parte del forward durante il backward. Distributed data parallelism replica il modello, mentre sharding di parametri, gradienti e optimizer state diventa necessario per backbone molto grandi. Queste tecniche cambiano prestazioni e memoria, non l'obiettivo matematico, ma possono alterare l'ordine delle riduzioni e la riproducibilità bitwise.

## Condizionamento e dropout

Per la classifier-free guidance, la condizione viene sostituita con un token nullo con probabilità $p_{\mathrm{drop}}$. Un dropout troppo basso produce un ramo unconditional debole; uno troppo alto riduce la capacità condizionale. Il token nullo deve essere costruito nello stesso modo in training e inferenza.

Con più condizioni — testo, immagini, stato propriocettivo — occorre decidere se rimuoverle congiuntamente o indipendentemente. La scelta determina quali combinazioni siano disponibili per guidance e ablation.

## Sampler e numero di valutazioni

La velocità non va misurata soltanto in passi. Un passo Heun usa normalmente due valutazioni del campo, mentre Eulero ne usa una. La metrica confrontabile è il numero di **network function evaluations (NFE)** insieme alla latenza effettiva.

Griglie uniformi in $t$, $\sigma$ o log-SNR non coincidono. Solver multistep riusano valutazioni precedenti; solver adattivi scelgono dinamicamente i passi ma producono latenza variabile. Nei sistemi robotici sono spesso preferibili budget fissi e profiling end-to-end, includendo encoder visuale, trasferimenti di memoria e post-processing delle azioni.

## Validazione e diagnostica

Una validazione utile separa almeno:

- loss per regione di rumore;
- qualità con pesi raw ed EMA;
- prestazione al variare di sampler e NFE;
- diversità tra seed a condizione fissa;
- sensibilità alla guidance;
- errori di ricostruzione dell'autoencoder, se presente.

Per policy robotiche, la MSE offline non sostituisce la valutazione closed loop. Vanno misurati successo, recovery, smoothness, violazioni dei limiti e latenza. Più campioni per osservazione possono stimare la multimodalità, ma scegliere retrospettivamente il migliore sovrastima la policy realmente deployabile.

## Riproducibilità

Un checkpoint è interpretabile soltanto insieme a configurazione e preprocessing. È necessario registrare almeno dataset e filtri, normalizzazione, schedule, target, loss weighting, distribuzione dei timestep, architettura, optimizer, batch size globale, EMA, precisione, sampler e seed.

Nel checkpoint dovrebbero essere inclusi modello, EMA, optimizer, scheduler del learning rate, scaler AMP e step globale. Un resume che ripristina soltanto i pesi cambia la traiettoria di training e non è equivalente alla continuazione dell'esperimento.

La regola pratica è trattare **training objective e sampling procedure come due configurazioni collegate ma distinte**. Molti errori attribuiti al modello derivano in realtà da una conversione incoerente tra target e sampler, da uno scaling latente errato o da una convenzione temporale invertita.
