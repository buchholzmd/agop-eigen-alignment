def check(name, err, tol, higher_is_better=False):
    """Record and print a numeric check.  err is the measured quantity, tol the bound."""
    ok = (err >= tol) if higher_is_better else (err <= tol)
    rel = ">=" if higher_is_better else "<="
    print(f"  [{'PASS' if ok else 'FAIL'}] {name:52s} {err:11.3e} {rel} {tol:.0e}")
    return ok