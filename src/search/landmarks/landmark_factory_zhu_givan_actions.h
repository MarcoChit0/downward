#ifndef LANDMARKS_LANDMARK_FACTORY_ZHU_GIVAN_ACTIONS_H
#define LANDMARKS_LANDMARK_FACTORY_ZHU_GIVAN_ACTIONS_H

#include "landmark_factory_relaxation.h"

#include "../utils/hash.h"

#include <unordered_set>
#include <vector>

namespace landmarks {
/*
  Action-landmark variant of the Zhu/Givan label propagation.

  This factory is intentionally SELF-CONTAINED: it does not modify, subclass, or
  otherwise depend on `LandmarkFactoryZhuGivan` (whose propagation machinery is
  all private and therefore not reusable from the outside). It only builds on
  the public `LandmarkFactoryRelaxation` framework, exactly like the fact-based
  factory does, so the existing Zhu/Givan code keeps working untouched.

  Difference from the fact-based factory: every proposition node carries a set
  of *operator ids* -- the actions that must occur in any relaxed plan that
  reaches that proposition -- instead of a set of facts. The action-landmark
  labels propagate independently of the fact labels, with the same AND/OR
  fixpoint as Zhu/Givan:
    - on reaching an operator (AND node): {op} union the labels of its
      preconditions;
    - on reaching a proposition (OR node): the intersection over the labels of
      its achievers.
  The action landmarks of the task are the union of the operator labels on the
  goal atoms (equivalently, the labels of the virtual FINISH action). They are
  written to stdout in a parseable <action-landmarks> ... </action-landmarks>
  block, mirroring the pyperplan reference implementation.

  Efficiency: like the fact-based factory -- and unlike the pyperplan reference,
  which rescans every node on every pass -- propagation is trigger-based. When a
  proposition is reached or its labels shrink, only the operators that depend on
  it are re-examined, so the fixpoint is reached without rescanning the whole
  graph each round.
*/
class LandmarkFactoryZhuGivanActions : public LandmarkFactoryRelaxation {
    // Operator/axiom ids; plain ints, so std::unordered_set suffices.
    using OpLabelSet = std::unordered_set<int>;

    struct PropNode {
        bool reached = false;
        // Operator ids that are landmarks for reaching this proposition.
        OpLabelSet labels;
    };
    // Indexed [var][value].
    using PropLayer = std::vector<std::vector<PropNode>>;

    /* triggers[var][value] lists the operators to re-examine once proposition
       (var, value) is reached or its labels change. */
    std::vector<std::vector<std::vector<int>>> triggers;
    std::vector<int> operators_without_preconditions;

    void compute_triggers(const TaskProxy &task_proxy);
    void add_operator_to_triggers(const OperatorProxy &op);

    static bool operator_is_applicable(
        const OperatorProxy &op, const PropLayer &layer);
    static bool conditions_hold(
        const ConditionsProxy &conditions, const PropLayer &layer);
    static OpLabelSet union_of_condition_labels(
        const ConditionsProxy &conditions, const PropLayer &layer);
    /* Merge `contribution` into `node`: seed on first reach, intersect after.
       Returns whether the labels changed. */
    static bool merge_action_labels(
        PropNode &node, const OpLabelSet &contribution);

    /* Apply `op` to `current` and propagate its action labels onto the effects
       it reaches in `next`. Returns the propositions whose labels changed or
       that were reached for the first time. */
    utils::HashSet<FactPair> apply_operator_and_propagate_labels(
        const OperatorProxy &op, const PropLayer &current,
        PropLayer &next) const;

    PropLayer initialize_layer(
        const TaskProxy &task_proxy,
        std::unordered_set<int> &triggered_ops) const;
    void propagate_until_fixpoint(
        const TaskProxy &task_proxy, std::unordered_set<int> &&triggered_ops,
        PropLayer &current) const;
    PropLayer build_labeled_graph(const TaskProxy &task_proxy) const;

    OpLabelSet collect_goal_action_landmarks(
        const TaskProxy &task_proxy, const PropLayer &layer) const;
    OpLabelSet action_label_set_of_operator(
        const OperatorProxy &op, const PropLayer &layer) const;
    void dump_action_landmarks(
        const TaskProxy &task_proxy, const PropLayer &layer,
        const OpLabelSet &goal_action_landmarks) const;
    // Keep the landmark graph valid (goals are always landmarks) so that any
    // heuristic using this factory still behaves correctly.
    void add_goal_atoms_as_landmarks(const TaskProxy &task_proxy) const;

    virtual void generate_relaxed_landmarks(
        const std::shared_ptr<AbstractTask> &task,
        Exploration &exploration) override;

public:
    explicit LandmarkFactoryZhuGivanActions(utils::Verbosity verbosity);

    virtual bool supports_conditional_effects() const override;
};
}

#endif
