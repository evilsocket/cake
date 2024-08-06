pub struct SafeScheduler<T> {
    pub(crate) scheduler: T,
}

unsafe impl<T> Send for SafeScheduler<T> {}
