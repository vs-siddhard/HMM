/**
 * Hidden Markov Model (HMM) Engine with Gaussian Acoustic Emissions (GMM-HMM)
 * Implements Forward, Backward, Viterbi Trellis decoding, Baum-Welch (EM) training,
 * and Left-to-Right speech acoustic modeling.
 */

export interface HMMTrainingEpoch {
  epoch: number;
  totalLogLikelihood: number;
  delta: number;
}

export interface ViterbiResult {
  bestScore: number;
  bestPath: number[];           // Array of state indices [q_1, q_2, ... q_T]
  trellis: number[][];          // [timeFrame][stateIndex] log probabilities
  backpointers: number[][];     // [timeFrame][stateIndex]
}

export interface WordScore {
  word: string;
  logLikelihood: number;
  posterior: number;            // Normalized confidence (0 - 1)
  bestPath: number[];
  numStates: number;
}

export interface RecognitionResult {
  bestWord: string;
  confidence: number;
  scores: WordScore[];
  inputFramesCount: number;
}

/**
 * Numerically stable Log-Sum-Exp trick:
 * log(sum(exp(x_i))) = m + log(sum(exp(x_i - m))) where m = max(x_i)
 */
export function logSumExp(values: number[]): number {
  if (values.length === 0) return -Infinity;
  let max = -Infinity;
  for (let i = 0; i < values.length; i++) {
    if (values[i] > max) max = values[i];
  }
  if (!isFinite(max)) return -Infinity;

  let sumExp = 0;
  for (let i = 0; i < values.length; i++) {
    sumExp += Math.exp(values[i] - max);
  }
  return max + Math.log(sumExp);
}

/**
 * Log Probability Density Function for Diagonal Gaussian Distribution:
 * log N(x; mu, sigma^2) = -0.5 * [ D * log(2*pi) + sum_d (log(sigma_d^2) + (x_d - mu_d)^2 / sigma_d^2) ]
 */
export function logGaussianDensity(x: number[], mean: number[], variance: number[]): number {
  const D = x.length;
  let sum = 0;
  const log2Pi = 1.8378770664093453; // log(2 * pi)

  for (let d = 0; d < D; d++) {
    const v = Math.max(variance[d], 1e-4); // Variance floor to avoid singularity
    const diff = x[d] - mean[d];
    sum += Math.log(v) + (diff * diff) / v;
  }

  return -0.5 * (D * log2Pi + sum);
}

export class GaussianHMM {
  public word: string;
  public nStates: number;
  public nFeatures: number;
  public startProb: number[];    // pi: initial state distribution (length nStates)
  public transMat: number[][];   // A: transition probability matrix [nStates x nStates]
  public means: number[][];      // mu: [nStates x nFeatures]
  public vars: number[][];       // sigma^2: [nStates x nFeatures]
  public isTrained: boolean = false;
  public trainingHistory: HMMTrainingEpoch[] = [];

  constructor(word: string, nStates = 4, nFeatures = 13) {
    this.word = word;
    this.nStates = nStates;
    this.nFeatures = nFeatures;

    // Strict Left-to-Right topology initialization (Bakis model for speech)
    // Start strictly at State 0
    this.startProb = new Array(nStates).fill(0);
    this.startProb[0] = 1.0;

    // Left-to-Right transition matrix: self-loop (e.g. 0.65) and forward transition (0.35)
    this.transMat = Array.from({ length: nStates }, (_, i) => {
      const row = new Array(nStates).fill(0);
      if (i === nStates - 1) {
        row[i] = 1.0; // Terminal state stays in self
      } else {
        row[i] = 0.65;
        row[i + 1] = 0.35;
      }
      return row;
    });

    this.means = Array.from({ length: nStates }, () => new Array(nFeatures).fill(0));
    this.vars = Array.from({ length: nStates }, () => new Array(nFeatures).fill(1.0));
  }

  /**
   * Log emission probability: log b_j(x_t)
   */
  public logEmissionProb(stateIndex: number, obs: number[]): number {
    return logGaussianDensity(obs, this.means[stateIndex], this.vars[stateIndex]);
  }

  /**
   * Forward Algorithm in Log Space:
   * Computes alpha_t(j) = log P(O_1...O_t, q_t = j | lambda)
   * Returns matrix alpha [T x nStates] and total log likelihood log P(O | lambda)
   */
  public forward(observations: number[][]): { alpha: number[][]; logLikelihood: number } {
    const T = observations.length;
    const N = this.nStates;
    const alpha: number[][] = Array.from({ length: T }, () => new Array(N).fill(-Infinity));

    // t = 0
    for (let j = 0; j < N; j++) {
      if (this.startProb[j] > 0) {
        alpha[0][j] = Math.log(this.startProb[j]) + this.logEmissionProb(j, observations[0]);
      }
    }

    // t = 1 to T - 1
    for (let t = 1; t < T; t++) {
      for (let j = 0; j < N; j++) {
        const emitLog = this.logEmissionProb(j, observations[t]);
        const transTerms: number[] = [];

        for (let i = 0; i < N; i++) {
          if (this.transMat[i][j] > 0 && isFinite(alpha[t - 1][i])) {
            transTerms.push(alpha[t - 1][i] + Math.log(this.transMat[i][j]));
          }
        }

        if (transTerms.length > 0) {
          alpha[t][j] = emitLog + logSumExp(transTerms);
        }
      }
    }

    // Total log-likelihood is logSumExp(alpha[T-1])
    const logLikelihood = logSumExp(alpha[T - 1]);
    return { alpha, logLikelihood };
  }

  /**
   * Backward Algorithm in Log Space:
   * Computes beta_t(i) = log P(O_{t+1}...O_T | q_t = i, lambda)
   */
  public backward(observations: number[][]): number[][] {
    const T = observations.length;
    const N = this.nStates;
    const beta: number[][] = Array.from({ length: T }, () => new Array(N).fill(-Infinity));

    // t = T - 1 (initialization)
    for (let i = 0; i < N; i++) {
      beta[T - 1][i] = 0; // log(1) = 0
    }

    // t = T - 2 down to 0
    for (let t = T - 2; t >= 0; t--) {
      for (let i = 0; i < N; i++) {
        const terms: number[] = [];
        for (let j = 0; j < N; j++) {
          if (this.transMat[i][j] > 0 && isFinite(beta[t + 1][j])) {
            const emitLog = this.logEmissionProb(j, observations[t + 1]);
            terms.push(Math.log(this.transMat[i][j]) + emitLog + beta[t + 1][j]);
          }
        }
        if (terms.length > 0) {
          beta[t][i] = logSumExp(terms);
        }
      }
    }

    return beta;
  }

  /**
   * Viterbi Decoding Algorithm:
   * Computes the single most likely hidden state sequence q_1*, ..., q_T*
   */
  public viterbi(observations: number[][]): ViterbiResult {
    const T = observations.length;
    const N = this.nStates;

    const trellis: number[][] = Array.from({ length: T }, () => new Array(N).fill(-Infinity));
    const backpointers: number[][] = Array.from({ length: T }, () => new Array(N).fill(0));

    // t = 0
    for (let j = 0; j < N; j++) {
      if (this.startProb[j] > 0) {
        trellis[0][j] = Math.log(this.startProb[j]) + this.logEmissionProb(j, observations[0]);
      }
    }

    // t = 1 to T - 1
    for (let t = 1; t < T; t++) {
      for (let j = 0; j < N; j++) {
        let maxVal = -Infinity;
        let bestPrev = 0;

        for (let i = 0; i < N; i++) {
          if (this.transMat[i][j] > 0 && isFinite(trellis[t - 1][i])) {
            const score = trellis[t - 1][i] + Math.log(this.transMat[i][j]);
            if (score > maxVal) {
              maxVal = score;
              bestPrev = i;
            }
          }
        }

        if (isFinite(maxVal)) {
          trellis[t][j] = maxVal + this.logEmissionProb(j, observations[t]);
          backpointers[t][j] = bestPrev;
        }
      }
    }

    // Find best final state at T - 1
    let bestScore = -Infinity;
    let bestFinalState = 0;
    for (let j = 0; j < N; j++) {
      if (trellis[T - 1][j] > bestScore) {
        bestScore = trellis[T - 1][j];
        bestFinalState = j;
      }
    }

    // Backtrack path
    const bestPath = new Array<number>(T);
    bestPath[T - 1] = bestFinalState;
    for (let t = T - 2; t >= 0; t--) {
      bestPath[t] = backpointers[t + 1][bestPath[t + 1]];
    }

    return {
      bestScore,
      bestPath,
      trellis,
      backpointers,
    };
  }

  /**
   * Score an observation sequence: returns the log-likelihood log P(O | lambda)
   */
  public score(observations: number[][]): number {
    if (observations.length === 0) return -Infinity;
    const { logLikelihood } = this.forward(observations);
    return logLikelihood;
  }

  /**
   * Fast Segmental Partition Training:
   * Divides each utterance into N contiguous temporal segments and calculates empirical means & variances.
   * Provides rapid convergence and excellent initialization for speech HMMs.
   */
  public fitSegmental(dataList: number[][][]): void {
    if (dataList.length === 0) return;
    const N = this.nStates;
    const D = this.nFeatures;

    const stateBuckets: number[][][] = Array.from({ length: N }, () => []);

    for (const sequence of dataList) {
      const T = sequence.length;
      if (T === 0) continue;

      for (let t = 0; t < T; t++) {
        // Uniform temporal partition
        const stateIdx = Math.min(N - 1, Math.floor((t / T) * N));
        stateBuckets[stateIdx].push(sequence[t]);
      }
    }

    // Compute means and variances for each state
    for (let j = 0; j < N; j++) {
      const bucket = stateBuckets[j];
      if (bucket.length === 0) continue;

      for (let d = 0; d < D; d++) {
        let sum = 0;
        for (let i = 0; i < bucket.length; i++) {
          sum += bucket[i][d];
        }
        const mean = sum / bucket.length;
        this.means[j][d] = mean;

        let varSum = 0;
        for (let i = 0; i < bucket.length; i++) {
          const diff = bucket[i][d] - mean;
          varSum += diff * diff;
        }
        // Variance floor
        this.vars[j][d] = Math.max(varSum / bucket.length, 1e-4);
      }
    }

    // Default left-to-right transitions
    for (let i = 0; i < N; i++) {
      for (let j = 0; j < N; j++) {
        if (i === N - 1) {
          this.transMat[i][j] = j === i ? 1.0 : 0.0;
        } else if (j === i) {
          this.transMat[i][j] = 0.65;
        } else if (j === i + 1) {
          this.transMat[i][j] = 0.35;
        } else {
          this.transMat[i][j] = 0.0;
        }
      }
    }

    this.isTrained = true;
  }

  /**
   * Full Baum-Welch (EM) Training Algorithm across multiple observation sequences
   */
  public fitBaumWelch(dataList: number[][][], maxIters = 15, tol = 1e-3): HMMTrainingEpoch[] {
    if (dataList.length === 0) return [];
    const N = this.nStates;
    const D = this.nFeatures;

    // First do segmental initialization to ensure meaningful initial Gaussian centers
    this.fitSegmental(dataList);

    this.trainingHistory = [];
    let prevLogLikelihood = -Infinity;

    for (let iter = 1; iter <= maxIters; iter++) {
      let totalLogLikelihood = 0;

      // Accumulators for M-Step
      const transNumerator: number[][] = Array.from({ length: N }, () => new Array(N).fill(0));
      const transDenominator: number[] = new Array(N).fill(0);

      const meanNumerator: number[][] = Array.from({ length: N }, () => new Array(D).fill(0));
      const varNumerator: number[][] = Array.from({ length: N }, () => new Array(D).fill(0));
      const gammaSum: number[] = new Array(N).fill(0);

      for (const sequence of dataList) {
        const T = sequence.length;
        if (T < 2) continue;

        const { alpha, logLikelihood } = this.forward(sequence);
        const beta = this.backward(sequence);

        if (!isFinite(logLikelihood)) continue;
        totalLogLikelihood += logLikelihood;

        // Compute gammas and xi
        // gamma[t][i] = P(q_t = S_i | O, lambda)
        const gamma: number[][] = Array.from({ length: T }, () => new Array(N).fill(0));
        for (let t = 0; t < T; t++) {
          for (let i = 0; i < N; i++) {
            const logGamma = alpha[t][i] + beta[t][i] - logLikelihood;
            gamma[t][i] = Math.exp(logGamma);
            gammaSum[i] += gamma[t][i];

            // Mean and Var accumulators
            for (let d = 0; d < D; d++) {
              meanNumerator[i][d] += gamma[t][i] * sequence[t][d];
            }
          }
        }

        // Xi transitions for t = 0 ... T - 2
        for (let t = 0; t < T - 1; t++) {
          for (let i = 0; i < N; i++) {
            transDenominator[i] += gamma[t][i];
            for (let j = 0; j < N; j++) {
              if (this.transMat[i][j] > 0) {
                const logXi =
                  alpha[t][i] +
                  Math.log(this.transMat[i][j]) +
                  this.logEmissionProb(j, sequence[t + 1]) +
                  beta[t + 1][j] -
                  logLikelihood;
                transNumerator[i][j] += Math.exp(logXi);
              }
            }
          }
        }
      }

      // M-Step 1: Update Means
      for (let i = 0; i < N; i++) {
        if (gammaSum[i] > 1e-6) {
          for (let d = 0; d < D; d++) {
            this.means[i][d] = meanNumerator[i][d] / gammaSum[i];
          }
        }
      }

      // M-Step 2: Update Variances (second pass over sequences with updated means)
      for (const sequence of dataList) {
        const T = sequence.length;
        if (T < 2) continue;
        const { alpha, logLikelihood } = this.forward(sequence);
        const beta = this.backward(sequence);
        if (!isFinite(logLikelihood)) continue;

        for (let t = 0; t < T; t++) {
          for (let i = 0; i < N; i++) {
            const g = Math.exp(alpha[t][i] + beta[t][i] - logLikelihood);
            for (let d = 0; d < D; d++) {
              const diff = sequence[t][d] - this.means[i][d];
              varNumerator[i][d] += g * diff * diff;
            }
          }
        }
      }

      for (let i = 0; i < N; i++) {
        if (gammaSum[i] > 1e-6) {
          for (let d = 0; d < D; d++) {
            this.vars[i][d] = Math.max(varNumerator[i][d] / gammaSum[i], 1e-4);
          }
        }
      }

      // M-Step 3: Update Left-to-Right Transition Matrix
      for (let i = 0; i < N; i++) {
        if (i === N - 1) {
          this.transMat[i] = new Array(N).fill(0);
          this.transMat[i][i] = 1.0;
        } else if (transDenominator[i] > 1e-6) {
          let rowSum = 0;
          for (let j = 0; j < N; j++) {
            // Strictly enforce Left-to-Right (only self and next state)
            if (j === i || j === i + 1) {
              this.transMat[i][j] = Math.max(0.001, transNumerator[i][j] / transDenominator[i]);
              rowSum += this.transMat[i][j];
            } else {
              this.transMat[i][j] = 0;
            }
          }
          if (rowSum > 0) {
            for (let j = 0; j < N; j++) {
              this.transMat[i][j] /= rowSum;
            }
          }
        }
      }

      const delta = Math.abs(totalLogLikelihood - prevLogLikelihood);
      this.trainingHistory.push({
        epoch: iter,
        totalLogLikelihood,
        delta,
      });

      if (iter > 1 && delta < tol) {
        break;
      }
      prevLogLikelihood = totalLogLikelihood;
    }

    this.isTrained = true;
    return this.trainingHistory;
  }
}

/**
 * Multi-Word Isolated Speech Recognizer
 */
export class SpeechRecognizer {
  public models: Map<string, GaussianHMM> = new Map();
  public nStatesPerWord: number;
  public nFeatures: number;

  constructor(words: string[] = ['apple', 'banana', 'cherry'], nStates = 4, nFeatures = 13) {
    this.nStatesPerWord = nStates;
    this.nFeatures = nFeatures;

    for (const word of words) {
      this.models.set(word, new GaussianHMM(word, nStates, nFeatures));
    }
  }

  public getModel(word: string): GaussianHMM | undefined {
    return this.models.get(word);
  }

  public addWord(word: string): GaussianHMM {
    const normalized = word.trim().toLowerCase();
    if (!this.models.has(normalized)) {
      const model = new GaussianHMM(normalized, this.nStatesPerWord, this.nFeatures);
      this.models.set(normalized, model);
      return model;
    }
    return this.models.get(normalized)!;
  }

  public removeWord(word: string): boolean {
    return this.models.delete(word.trim().toLowerCase());
  }

  public getAllWords(): string[] {
    return Array.from(this.models.keys());
  }

  /**
   * Train model for a word using list of observation sequences (MFCC feature arrays)
   */
  public trainWord(word: string, dataList: number[][][], useBaumWelch = true, iterations = 10): void {
    const model = this.addWord(word);
    if (useBaumWelch) {
      model.fitBaumWelch(dataList, iterations);
    } else {
      model.fitSegmental(dataList);
    }
  }

  /**
   * Predict / Recognize word for unknown observation sequence
   */
  public predict(observations: number[][]): RecognitionResult {
    const scores: WordScore[] = [];

    for (const [word, model] of this.models.entries()) {
      const viterbiRes = model.viterbi(observations);
      const logLikelihood = model.score(observations);

      scores.push({
        word,
        logLikelihood: isFinite(logLikelihood) ? logLikelihood : viterbiRes.bestScore,
        posterior: 0,
        bestPath: viterbiRes.bestPath,
        numStates: model.nStates,
      });
    }

    // Sort by log-likelihood descending
    scores.sort((a, b) => b.logLikelihood - a.logLikelihood);

    // Compute softmax posterior probabilities with temperature scaling
    // Scale down values to avoid sharp 1.0 vs 0.0 saturations
    const logScores = scores.map((s) => s.logLikelihood);
    const maxLog = Math.max(...logScores);
    const temperature = 40.0; // softens acoustic scale
    const expScores = logScores.map((s) => Math.exp((s - maxLog) / temperature));
    const sumExp = expScores.reduce((a, b) => a + b, 0);

    for (let i = 0; i < scores.length; i++) {
      scores[i].posterior = sumExp > 0 ? expScores[i] / sumExp : 1 / scores.length;
    }

    const bestWord = scores.length > 0 ? scores[0].word : '';
    const confidence = scores.length > 0 ? scores[0].posterior : 0;

    return {
      bestWord,
      confidence,
      scores,
      inputFramesCount: observations.length,
    };
  }
}
