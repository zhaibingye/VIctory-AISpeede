/// A union of nonempty half-open byte ranges, sorted and merged across adjacency.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct ByteRanges(Storage);

type Span = (usize, usize);

#[derive(Clone, Debug, Default, PartialEq, Eq)]
enum Storage {
    #[default]
    Empty,
    One(Span),
    Many(Vec<Span>),
}

impl ByteRanges {
    pub(crate) fn insert(&mut self, from: usize, to: usize) {
        if from >= to {
            return;
        }

        match &mut self.0 {
            Storage::Empty => self.0 = Storage::One((from, to)),
            Storage::One((start, end)) => {
                if to < *start {
                    self.0 = Storage::Many(vec![(from, to), (*start, *end)]);
                } else if *end < from {
                    self.0 = Storage::Many(vec![(*start, *end), (from, to)]);
                } else {
                    *start = (*start).min(from);
                    *end = (*end).max(to);
                }
            }
            Storage::Many(spans) => {
                // Ends increase with starts because the stored spans have gaps between them.
                let first = spans.partition_point(|&(_, end)| end < from);
                let past = first + spans[first..].partition_point(|&(start, _)| start <= to);
                if first == past {
                    spans.insert(first, (from, to));
                } else {
                    let joined = (from.min(spans[first].0), to.max(spans[past - 1].1));
                    if first == 0 && past == spans.len() {
                        self.0 = Storage::One(joined);
                    } else {
                        spans[first] = joined;
                        if first + 1 < past {
                            drop(spans.drain(first + 1..past));
                        }
                    }
                }
            }
        }
    }

    fn as_slice(&self) -> &[Span] {
        match &self.0 {
            Storage::Empty => &[],
            Storage::One(span) => std::slice::from_ref(span),
            Storage::Many(spans) => spans,
        }
    }

    pub(crate) fn iter(&self) -> impl Iterator<Item = (usize, usize)> + '_ {
        self.as_slice().iter().copied()
    }

    pub(crate) fn contains(&self, from: usize, to: usize) -> bool {
        if from >= to {
            return false;
        }

        let spans = self.as_slice();
        let i = spans.partition_point(|&(_, end)| end <= from);
        spans.get(i).is_some_and(|&(start, end)| start <= from && to <= end)
    }

    pub(crate) fn byte_len(&self) -> usize {
        self.iter().fold(0usize, |len, (from, to)| len.saturating_add(to - from))
    }

    pub(crate) fn bounds(&self) -> Option<(usize, usize)> {
        let spans = self.as_slice();
        Some((spans.first()?.0, spans.last()?.1))
    }
}

#[cfg(test)]
mod tests {
    use super::ByteRanges;

    #[test]
    fn merges_overlap_and_adjacency_without_filling_other_gaps() {
        let mut ranges = ByteRanges::default();
        for (from, to) in [(20, 24), (2, 5), (30, 32), (8, 12), (5, 9), (12, 16), (4, 7)] {
            ranges.insert(from, to);
        }
        assert_eq!(ranges.iter().collect::<Vec<_>>(), [(2, 16), (20, 24), (30, 32)]);
        assert_eq!(ranges.byte_len(), 20);

        ranges.insert(16, 30);
        assert_eq!(ranges.iter().collect::<Vec<_>>(), [(2, 32)]);
        assert!(ranges.contains(2, 32));
        assert_eq!(ranges.byte_len(), 30);
    }

    #[test]
    fn shared_bounds_do_not_imply_coverage() {
        let mut sparse = ByteRanges::default();
        sparse.insert(10, 20);
        sparse.insert(21, 30);
        let mut full = sparse.clone();
        full.insert(20, 21);
        assert_eq!(sparse.bounds(), Some((10, 30)));
        assert_eq!(sparse.bounds(), full.bounds());
        assert_eq!(sparse.byte_len(), 19);
        assert_eq!(full.byte_len(), 20);
        for (from, to) in [(10, 30), (19, 22), (20, 21)] {
            assert!(!sparse.contains(from, to));
            assert!(full.contains(from, to));
        }
        assert!(sparse.contains(10, 20));
        assert!(sparse.contains(21, 30));
    }

    #[test]
    fn ignores_empty_and_reversed_ranges() {
        let mut ranges = ByteRanges::default();
        ranges.insert(3, 3);
        ranges.insert(5, 1);
        assert_eq!(ranges.iter().next(), None);
        assert_eq!(ranges.bounds(), None);
        assert_eq!(ranges.byte_len(), 0);

        ranges.insert(2, 6);
        assert!(!ranges.contains(3, 3));
        assert!(!ranges.contains(5, 3));
    }

    #[test]
    fn handles_maximum_endpoint() {
        let max = usize::MAX;
        let mut ranges = ByteRanges::default();
        ranges.insert(max - 1, max);
        ranges.insert(0, max - 2);
        assert_eq!(ranges.byte_len(), max - 1);
        ranges.insert(max - 2, max - 1);
        assert_eq!(ranges.byte_len(), max);
        assert!(ranges.contains(0, max));
    }
}
