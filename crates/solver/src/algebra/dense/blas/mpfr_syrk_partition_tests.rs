use super::*;

fn leaves(n: usize, begin: usize, end: usize, lanes: usize, upper: bool) -> Vec<(usize, usize)> {
    if lanes == 1 {
        return vec![(begin, end)];
    }
    let cut = syrk_cut(n, begin, end, lanes, upper);
    let left = lanes / 2;
    assert!((begin + left..=end - (lanes - left)).contains(&cut));
    // Check the selected boundary against the actual touched-entry sum,
    // independently of the prefix formula and binary search.
    let weight = |a: usize, b: usize| {
        (a..b)
            .map(|j| if upper { j + 1 } else { n - j })
            .sum::<usize>()
    };
    let error = |at: usize| (weight(begin, at) * lanes).abs_diff(weight(begin, end) * left);
    assert_eq!(
        error(cut),
        (begin + left..=end - (lanes - left))
            .map(error)
            .min()
            .unwrap()
    );
    let mut out = leaves(n, begin, cut, left, upper);
    out.extend(leaves(n, cut, end, lanes - left, upper));
    out
}
#[test]
fn triangular_partitions_preserve_budget_and_balance() {
    for n in 1usize..=65 {
        for upper in [false, true] {
            for lanes in 1..=n.min(16) {
                let spans = leaves(n, 0, n, lanes, upper);
                assert_eq!(spans.len(), lanes);
                assert_eq!(spans[0].0, 0);
                assert_eq!(spans.last().unwrap().1, n);
                assert!(spans.iter().all(|&(a, b)| a < b));
                assert!(spans.windows(2).all(|w| w[0].1 == w[1].0));
            }
            // Representative eight-task triangle: equal columns have a
            // heaviest lane of 69 entries; weighted splits cap it at 45.
            if n == 24 {
                let spans = leaves(n, 0, n, 8, upper);
                let heaviest = spans
                    .iter()
                    .map(|&(a, b)| {
                        (a..b)
                            .map(|j| if upper { j + 1 } else { n - j })
                            .sum::<usize>()
                    })
                    .max()
                    .unwrap();
                assert!(heaviest <= 45);
            }
        }
    }
}
