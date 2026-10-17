// Hide & Seek question engine. Mirrors hs_core.Engine (Python) one-to-one; test_smoke.py runs both and compares.
//  simulate(secret)      : the strict engine: truthful answers, hard elimination, greedy information gain (the kill-test metric).
//  nextStep(hist, wrong) : the page protocol.  Phase A = hard elimination, ends with "Is it this one?".  If that guess is wrong,
//                          or the answers contradict every clip, phase B = soft posterior (each contradicting answer costs a
//                          factor eps) keeps asking the most informative question and guesses again (max_guesses rounds).
//  playProtocol(secret, flipAt) : a judge answering from the index attributes, optionally flipping the answers at given positions.
// hist = [{qi, a}] (a: true | false | null for "not sure"), wrong = [[clip ids of a rejected guess], ...]
function makeEngine(DATA) {
  const C = DATA.clips, Q = DATA.questions, N = C.length, PARAMS = DATA.params;
  const H = (k, n) => {
    if (k <= 0 || k >= n) return 0;
    const p = k / n;
    return -(p * Math.log2(p) + (1 - p) * Math.log2(1 - p));
  };
  function best(cand, asked) {
    let b = 1e-9, bi = -1;
    for (let qi = 0; qi < Q.length; qi++) {
      if (asked.has(qi)) continue;
      let k = 0;
      for (const i of cand) if (Q[qi].ans[i]) k++;
      const g = H(k, cand.length);
      if (g > b + 1e-12) { b = g; bi = qi; }
    }
    return bi;
  }
  function simulate(secret) {  // truthful answers; returns {n, final}
    let cand = [...Array(N).keys()], asked = new Set(), n = 0;
    while (cand.length > 1 && n < 40) {
      const qi = best(cand, asked);
      if (qi < 0) break;
      const a = Q[qi].ans[secret];
      asked.add(qi); n++;
      cand = cand.filter(i => Q[qi].ans[i] === a);
    }
    return {n, final: cand};
  }
  function hardCand(hist) {
    let cand = [...Array(N).keys()];
    for (const h of hist) if (h.a !== null) cand = cand.filter(i => Q[h.qi].ans[i] === h.a);
    return cand;
  }
  function weights(hist, rejected, eps) {
    const w = [];
    for (let i = 0; i < N; i++) {
      if (rejected.has(i)) { w.push(0); continue; }
      let m = 0;
      for (const h of hist) if (h.a !== null && Q[h.qi].ans[i] !== h.a) m++;
      w.push(Math.pow(eps, m));
    }
    return w;
  }
  function nextStep(hist, wrong, over) {
    const P = Object.assign({}, PARAMS, over || {});
    const asked = new Set(hist.map(h => h.qi));
    const rejected = new Set();
    wrong.forEach(g => g.forEach(i => rejected.add(i)));
    if (wrong.length >= P.max_guesses) return {type: 'giveup'};
    const hard = hardCand(hist);
    if (!rejected.size && hard.length) {
      if (hard.length === 1) return {type: 'guess', ids: hard};
      const qi = best(hard, asked);
      if (qi < 0) return {type: 'guess', ids: hard.slice(0, P.max_show)};
      return {type: 'ask', qi};
    }
    const w = weights(hist, rejected, P.eps);
    let tot = 0; for (const x of w) tot += x;
    if (tot <= 0) return {type: 'giveup'};
    const post = w.map(x => x / tot);
    let top = 0; for (const x of post) if (x > top) top = x;
    const group = []; for (let i = 0; i < N; i++) if (post[i] >= top - 1e-9) group.push(i);
    const grp = group.slice(0, P.max_show);
    if (top >= P.stop || asked.size >= P.max_questions) return {type: 'guess', ids: grp};
    const eps = P.eps, he = -(eps * Math.log2(eps) + (1 - eps) * Math.log2(1 - eps));
    let b = 1e-6, bi = -1;
    for (let qi = 0; qi < Q.length; qi++) {
      if (asked.has(qi)) continue;
      let py = 0; for (let i = 0; i < N; i++) if (Q[qi].ans[i]) py += post[i];
      const pr = py * (1 - eps) + (1 - py) * eps;
      const hq = (pr <= 0 || pr >= 1) ? 0 : -(pr * Math.log2(pr) + (1 - pr) * Math.log2(1 - pr));
      const g = hq - he;
      if (g > b + 1e-12) { b = g; bi = qi; }
    }
    if (bi < 0) return {type: 'guess', ids: grp};
    return {type: 'ask', qi: bi};
  }
  function playProtocol(secret, flipAt) {
    const hist = [], wrong = [];
    let rounds = 0, first = null;
    const flips = new Set(flipAt || []);
    for (let guard = 0; guard < 200; guard++) {
      const st = nextStep(hist, wrong);
      if (st.type === 'ask') {
        let a = !!Q[st.qi].ans[secret];
        if (flips.has(hist.length)) a = !a;
        hist.push({qi: st.qi, a});
        continue;
      }
      if (st.type === 'giveup') return {found: false, n: hist.length, guesses: rounds, first};
      rounds++;
      if (first === null) first = st.ids.length;
      if (st.ids.includes(secret)) return {found: true, n: hist.length, guesses: rounds, first};
      wrong.push(st.ids.slice());
    }
    return {found: false, n: hist.length, guesses: rounds, first};
  }
  return {C, Q, N, H, best, simulate, hardCand, weights, nextStep, playProtocol, PARAMS};
}
if (typeof module !== 'undefined') module.exports = {makeEngine};
