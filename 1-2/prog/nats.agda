open import Data.Nat
import Relation.Binary.PropositionalEquality as Eq
open Eq using (_≡_; refl)
open Eq.≡-Reasoning using (begin_; _≡⟨⟩_; _∎)

_ : 3 + 4 ≡ 7
_ =
    begin
        3 + 4
    ≡⟨⟩
       suc (suc (suc zero)) + suc (suc (suc (suc zero)))
    ≡⟨⟩
        suc (suc (suc zero) + suc (suc (suc (suc zero))))
    ≡⟨⟩
        suc (suc (suc zero + suc (suc (suc (suc zero)))))
    ≡⟨⟩
        suc (suc (suc (zero + suc (suc (suc (suc zero))))))
    ≡⟨⟩
        suc (suc (suc (suc (suc (suc (suc zero))))))
    ∎


