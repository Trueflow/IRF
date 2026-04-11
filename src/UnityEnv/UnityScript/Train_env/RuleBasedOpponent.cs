using System.Collections.Generic;
using UnityEngine;

/// <summary>
/// Tactical rule-based opponent for the existing RSA environment.
/// Attach this component to every red-agent GameObject that already has an
/// RSAcontrol component. This component disables only RSAcontrol.FixedUpdate
/// (the uniformly random policy) and reuses RSAcontrol's combat, masking,
/// damage, death, and reset interfaces.
/// </summary>
[DisallowMultipleComponent]
[RequireComponent(typeof(RSAcontrol))]
public class RuleBasedOpponent : MonoBehaviour
{
    [Header("Decision policy")]
    [Tooltip("Prefer targets that can be eliminated quickly.")]
    [Range(0f, 10f)] public float eliminationPriority = 4f;

    [Tooltip("Prefer high-value unit types when two targets are otherwise similar.")]
    [Range(0f, 5f)] public float unitValuePriority = 1f;

    [Tooltip("Small distance penalty used when choosing among targets.")]
    [Range(0f, 1f)] public float distancePenalty = 0.05f;

    [Header("Tactical movement")]
    [Tooltip("Angular error tolerated before the opponent turns toward its target.")]
    [Range(1f, 45f)] public float aimTolerance = 8f;

    [Tooltip("If true, agents strafe while waiting for a firing opportunity.")]
    public bool strafeWhileHoldingRange = true;

    [Tooltip("Adds limited action noise for opponent diversity. Set to zero for deterministic rules.")]
    [Range(0f, 0.25f)] public float actionNoiseProbability = 0.02f;

    [Header("Strength control")]
    [Tooltip("Select a new action once per N physics steps. Shooting cooldown still decreases every physics step.")]
    [Range(1, 20)] public int decisionInterval = 6;

    [Tooltip("Probability of passing up an otherwise feasible shot.")]
    [Range(0f, 0.9f)] public float attackSkipProbability = 0.45f;

    [Tooltip("Keep the selected target for this many physics steps unless it dies.")]
    [Range(0, 200)] public int targetLockSteps = 50;

    private RSAcontrol rsa;
    private Rigidbody body;
    private int decisionWait;
    private int targetLockRemaining;
    private rsa_MARLagent lockedTarget;

    private void Awake()
    {
        rsa = GetComponent<RSAcontrol>();
        body = GetComponent<Rigidbody>();
    }

    private void Start()
    {
        // RSAcontrol.Awake has already initialized unit statistics. Disabling
        // the component prevents its random FixedUpdate policy from running;
        // its public combat and masking methods remain callable.
        rsa.enabled = false;
    }

    private void FixedUpdate()
    {
        if (rsa == null || !rsa.IsActive)
        {
            return;
        }

        // Must run every physics step: getAttackMask() decreases
        // remainShootingCool by exactly one, preserving ShootingCooldownTerm.
        rsa.GetActionMask();

        if (targetLockRemaining > 0)
        {
            targetLockRemaining--;
        }

        // Slow tactical decisions only. Cooldown processing above remains
        // step-based and is not multiplied by decisionInterval.
        if (decisionWait > 0)
        {
            decisionWait--;
            HoldPosition();
            EnforcePhysicalConstraints();
            return;
        }
        decisionWait = Mathf.Max(1, decisionInterval) - 1;

        rsa_MARLagent strategicTarget = GetLockedStrategicTarget();
        if (strategicTarget == null)
        {
            HoldPosition();
            EnforcePhysicalConstraints();
            return;
        }

        // A feasible shot has priority over movement. Among feasible targets,
        // select the target requiring the fewest hits, then prefer valuable and
        // nearby units. This produces coordinated focus fire without sharing
        // hidden state beyond the information already used by RSAcontrol.
        if (Random.value >= attackSkipProbability
            && TryAttackPreferredTarget(strategicTarget))
        {
            EnforcePhysicalConstraints();
            return;
        }

        ExecuteTacticalMovement(strategicTarget);
        EnforcePhysicalConstraints();
    }

    private rsa_MARLagent GetLockedStrategicTarget()
    {
        bool lockIsValid = lockedTarget != null
            && lockedTarget.IsActive
            && lockedTarget.hp > 0f;

        if (lockIsValid && targetLockRemaining > 0)
        {
            return lockedTarget;
        }

        lockedTarget = SelectStrategicTarget();
        targetLockRemaining = lockedTarget == null ? 0 : Mathf.Max(0, targetLockSteps);
        return lockedTarget;
    }

    private rsa_MARLagent SelectStrategicTarget()
    {
        rsa_MARLagent best = null;
        float bestScore = float.NegativeInfinity;

        foreach (rsa_MARLagent candidate in rsa.count.AgentList)
        {
            if (candidate == null || !candidate.IsActive || candidate.hp <= 0f)
            {
                continue;
            }

            float distance = PlanarDistance(transform.position, candidate.transform.position);
            float hitsToKill = Mathf.Ceil(candidate.hp / Mathf.Max(rsa.attack, 0.001f));
            float value = Mathf.Max(candidate.rewardWeight, 0.1f);

            float score = eliminationPriority / Mathf.Max(hitsToKill, 1f)
                        + unitValuePriority * value
                        - distancePenalty * distance;

            // Stable tie-breaker: all red agents tend to agree on the same
            // target, creating focus fire instead of random target switching.
            score -= candidate.MyIndex * 0.0001f;

            if (score > bestScore)
            {
                bestScore = score;
                best = candidate;
            }
        }

        return best;
    }

    private bool TryAttackPreferredTarget(rsa_MARLagent preferredTarget)
    {
        if (rsa.ActionMask == null || rsa.ActionMask.Count <= 8)
        {
            return false;
        }

        if (rsa.count.AttackLeastDistance)
        {
            if (rsa.ActionMask[8] > 0f && rsa.target != null)
            {
                rsa.RSAShoot(rsa, rsa.target);
                return true;
            }
            return false;
        }

        if (preferredTarget == null)
        {
            return false;
        }

        int preferredIndex = rsa.count.AgentList.IndexOf(preferredTarget);
        int attackActionIndex = preferredIndex + 8;
        bool preferredShotIsFeasible = preferredIndex >= 0
            && attackActionIndex < rsa.ActionMask.Count
            && rsa.ActionMask[attackActionIndex] > 0f;

        if (!preferredShotIsFeasible)
        {
            return false;
        }

        rsa.target = preferredTarget.gameObject;
        rsa.RSAShoot(rsa, rsa.target);
        return true;
    }

    private void ExecuteTacticalMovement(rsa_MARLagent target)
    {
        Vector3 toTarget = target.transform.position - transform.position;
        toTarget.y = 0f;
        if (toTarget.sqrMagnitude < 0.0001f)
        {
            HoldPosition();
            return;
        }

        float signedAngle = Vector3.SignedAngle(transform.forward, toTarget.normalized, Vector3.up);
        float distance = toTarget.magnitude;
        float desiredRange = GetDesiredRange();

        // A small amount of bounded policy noise prevents a single perfectly
        // deterministic trajectory while retaining a clearly rule-based policy.
        if (actionNoiseProbability > 0f && Random.value < actionNoiseProbability)
        {
            ExecuteFirstFeasibleMovement((rsa.MyIndex % 2 == 0) ? 4 : 5);
            return;
        }

        if (Mathf.Abs(signedAngle) > aimTolerance)
        {
            RotateToward(signedAngle);
            return;
        }

        // Close the distance when outside effective range, withdraw when the
        // target is too close, and otherwise strafe to avoid remaining static.
        if (distance > desiredRange * 1.10f)
        {
            ExecuteFirstFeasibleMovement(2); // forward
        }
        else if (distance < desiredRange * 0.55f)
        {
            ExecuteFirstFeasibleMovement(3); // backward
        }
        else if (strafeWhileHoldingRange)
        {
            ExecuteFirstFeasibleMovement((rsa.MyIndex % 2 == 0) ? 4 : 5);
        }
        else
        {
            HoldPosition();
        }
    }

    private float GetDesiredRange()
    {
        switch (rsa.position)
        {
            case RSAcontrol.Position.Infantry:
                return rsa.ShootingRange * 0.75f;
            case RSAcontrol.Position.Armored:
                return rsa.ShootingRange * 0.80f;
            case RSAcontrol.Position.Tank:
                return rsa.ShootingRange * 0.88f;
            default:
                return rsa.ShootingRange * 0.80f;
        }
    }

    private void RotateToward(float signedAngle)
    {
        int preferredAction = signedAngle > 0f ? 6 : 7;
        if (!IsFeasible(preferredAction))
        {
            return;
        }

        float maxRotation = GetRotationStep();
        float rotation = Mathf.Clamp(signedAngle, -maxRotation, maxRotation);
        transform.Rotate(Vector3.up, rotation, Space.World);

        if (body != null)
        {
            body.angularVelocity = Vector3.zero;
        }
    }

    private void ExecuteFirstFeasibleMovement(int preferredAction)
    {
        int action = preferredAction;
        if (!IsFeasible(action))
        {
            // Prefer lateral displacement before retreating or stopping when
            // the intended path is blocked.
            int[] fallbacks = { 4, 5, 3, 2, 1 };
            action = 1;
            foreach (int candidate in fallbacks)
            {
                if (IsFeasible(candidate))
                {
                    action = candidate;
                    break;
                }
            }
        }

        Vector3 direction = Vector3.zero;
        switch (action)
        {
            case 2: direction = transform.forward; break;
            case 3: direction = -transform.forward; break;
            case 4: direction = transform.right; break;
            case 5: direction = -transform.right; break;
            default: HoldPosition(); return;
        }

        if (rsa.count.RandomNoise)
        {
            direction += new Vector3(Random.Range(-0.05f, 0.05f), 0f,
                                     Random.Range(-0.05f, 0.05f));
        }

        if (body != null)
        {
            body.AddForce(GetMoveSpeed() * direction, ForceMode.VelocityChange);
        }
    }

    private bool IsFeasible(int action)
    {
        return rsa.ActionMask != null
            && action >= 0
            && action < rsa.ActionMask.Count
            && rsa.ActionMask[action] > 0f;
    }

    private void HoldPosition()
    {
        if (body != null)
        {
            body.angularVelocity = Vector3.zero;
        }
    }

    private float GetMoveSpeed()
    {
        switch (rsa.position)
        {
            case RSAcontrol.Position.Infantry: return 1f;
            case RSAcontrol.Position.Armored: return 1.5f;
            case RSAcontrol.Position.Tank: return 1.25f;
            default: return 1f;
        }
    }

    private float GetRotationStep()
    {
        switch (rsa.position)
        {
            case RSAcontrol.Position.Infantry: return 5f;
            case RSAcontrol.Position.Armored: return 2.5f;
            case RSAcontrol.Position.Tank: return 2f;
            default: return 3f;
        }
    }

    private void EnforcePhysicalConstraints()
    {
        if (Mathf.Abs(transform.position.z) > rsa.env_threshold_z
            || Mathf.Abs(transform.position.x) > rsa.env_threshold_x
            || transform.position.y < -5f)
        {
            rsa.ActiveFalse();
            transform.SetPositionAndRotation(Vector3.zero, Quaternion.identity);
            return;
        }

        Vector3 angles = transform.rotation.eulerAngles;
        float xTilt = Mathf.Abs(Mathf.DeltaAngle(0f, angles.x));
        float zTilt = Mathf.Abs(Mathf.DeltaAngle(0f, angles.z));
        if (xTilt > 5f || zTilt > 5f)
        {
            transform.rotation = Quaternion.Euler(0f, angles.y, 0f);
        }
    }

    private static float PlanarDistance(Vector3 a, Vector3 b)
    {
        a.y = 0f;
        b.y = 0f;
        return Vector3.Distance(a, b);
    }
}
