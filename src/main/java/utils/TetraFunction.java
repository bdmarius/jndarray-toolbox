package utils;

import java.util.Objects;
import java.util.function.Function;

@FunctionalInterface
public interface TetraFunction<T, U, V, X, R> {

    R apply(T t, U u, V v, X x);

    default <K> TetraFunction<T, U, V, X, K> andThen(Function<? super R, ? extends K> after) {
        Objects.requireNonNull(after);
        return (T t, U u, V v, X x) -> after.apply(apply(t, u, v, x));
    }
}
