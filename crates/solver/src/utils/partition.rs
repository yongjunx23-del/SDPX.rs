/// Greedy contiguous split of weighted items into `parts` lanes, returning
/// each lane's first item. `prefix` holds running weights (`prefix[0] == 0`,
/// one more entry than items). Each lane takes items until it reaches an
/// equal share of the remaining weight, stepping back one item when that
/// lands closer to the share; the last lane takes the rest. Requires
/// `parts <= prefix.len() - 1`.
pub(crate) fn contiguous_lanes(prefix: &[u128], parts: usize) -> Vec<usize> {
    let items = prefix.len() - 1;
    let mut lanes = Vec::with_capacity(parts);
    let mut begin = 0;
    for lane in 0..parts {
        lanes.push(begin);
        let remaining = parts - lane;
        if remaining == 1 {
            break;
        }
        let target = (prefix[items] - prefix[begin]) / remaining as u128;
        let last = items - (remaining - 1);
        let mut end = begin + 1;
        while end < last && prefix[end] - prefix[begin] < target {
            end += 1;
        }
        if end > begin + 1
            && target.abs_diff(prefix[end - 1] - prefix[begin])
                <= target.abs_diff(prefix[end] - prefix[begin])
        {
            end -= 1;
        }
        begin = end;
    }
    lanes
}
