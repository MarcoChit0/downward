#include "landmark_factory_zhu_givan_actions.h"

#include "landmark.h"
#include "landmark_graph.h"
#include "util.h"

#include "../task_proxy.h"

#include "../plugins/plugin.h"
#include "../utils/logging.h"

#include <algorithm>
#include <iostream>
#include <unordered_set>
#include <vector>

using namespace std;

namespace landmarks {
LandmarkFactoryZhuGivanActions::LandmarkFactoryZhuGivanActions(
    utils::Verbosity verbosity)
    : LandmarkFactoryRelaxation(verbosity) {
}

void LandmarkFactoryZhuGivanActions::generate_relaxed_landmarks(
    const shared_ptr<AbstractTask> &task, Exploration &) {
    TaskProxy task_proxy(*task);
    if (log.is_at_least_normal()) {
        log << "Generating ACTION landmarks using Zhu/Givan label propagation"
            << endl;
    }

    compute_triggers(task_proxy);
    PropLayer last_layer = build_labeled_graph(task_proxy);
    OpLabelSet goal_action_landmarks =
        collect_goal_action_landmarks(task_proxy, last_layer);
    dump_action_landmarks(task_proxy, last_layer, goal_action_landmarks);

    /* Keep the landmark graph in a valid, sound state. We deliberately do not
       try to encode action landmarks as graph nodes (the graph is fact-based);
       we only register the goal atoms, which are always landmarks. */
    add_goal_atoms_as_landmarks(task_proxy);
}

// ── trigger bookkeeping (independent copy; original factory untouched) ────────
void LandmarkFactoryZhuGivanActions::compute_triggers(
    const TaskProxy &task_proxy) {
    assert(triggers.empty());
    const VariablesProxy &variables = task_proxy.get_variables();
    triggers.resize(variables.size());
    for (int i = 0; i < static_cast<int>(variables.size()); ++i) {
        triggers[i].resize(variables[i].get_domain_size());
    }
    for (OperatorProxy op : task_proxy.get_operators()) {
        add_operator_to_triggers(op);
    }
    for (OperatorProxy axiom : task_proxy.get_axioms()) {
        add_operator_to_triggers(axiom);
    }
}

void LandmarkFactoryZhuGivanActions::add_operator_to_triggers(
    const OperatorProxy &op) {
    int op_or_axiom_id = get_operator_or_axiom_id(op);
    const PreconditionsProxy &preconditions = op.get_preconditions();
    for (FactProxy precondition : preconditions) {
        auto [var, value] = precondition.get_pair();
        triggers[var][value].push_back(op_or_axiom_id);
    }
    for (EffectProxy effect : op.get_effects()) {
        for (FactProxy effect_condition : effect.get_conditions()) {
            auto [var, value] = effect_condition.get_pair();
            triggers[var][value].push_back(op_or_axiom_id);
        }
    }
    if (preconditions.empty()) {
        operators_without_preconditions.push_back(op_or_axiom_id);
    }
}

// ── label propagation ─────────────────────────────────────────────────────────
bool LandmarkFactoryZhuGivanActions::operator_is_applicable(
    const OperatorProxy &op, const PropLayer &layer) {
    for (FactProxy precondition : op.get_preconditions()) {
        auto [var, value] = precondition.get_pair();
        if (!layer[var][value].reached) {
            return false;
        }
    }
    return true;
}

bool LandmarkFactoryZhuGivanActions::conditions_hold(
    const ConditionsProxy &conditions, const PropLayer &layer) {
    for (FactProxy condition : conditions) {
        auto [var, value] = condition.get_pair();
        if (!layer[var][value].reached) {
            return false;
        }
    }
    return true;
}

LandmarkFactoryZhuGivanActions::OpLabelSet
LandmarkFactoryZhuGivanActions::union_of_condition_labels(
    const ConditionsProxy &conditions, const PropLayer &layer) {
    OpLabelSet result;
    for (FactProxy condition : conditions) {
        auto [var, value] = condition.get_pair();
        const OpLabelSet &labels = layer[var][value].labels;
        result.insert(labels.begin(), labels.end());
    }
    return result;
}

/*
  Merge `contribution` into the labels of a (just reached) proposition `node`.
  The first achiever to reach a proposition seeds its labels; every later
  achiever intersects, since an action is a landmark for the proposition only if
  *every* achiever needs it. Returns whether anything changed (first reach, or a
  shrinking intersection).
*/
bool LandmarkFactoryZhuGivanActions::merge_action_labels(
    PropNode &node, const OpLabelSet &contribution) {
    if (!node.reached) {
        node.reached = true;
        node.labels = contribution;
        return true;
    }
    size_t old_size = node.labels.size();
    if (old_size == 0) {
        // Already minimal: the proposition needs no action landmark.
        return false;
    }
    OpLabelSet intersection;
    const OpLabelSet &smaller =
        (node.labels.size() <= contribution.size()) ? node.labels : contribution;
    const OpLabelSet &larger =
        (node.labels.size() <= contribution.size()) ? contribution : node.labels;
    for (int label : smaller) {
        if (larger.count(label)) {
            intersection.insert(label);
        }
    }
    node.labels = move(intersection);
    return node.labels.size() != old_size;
}

utils::HashSet<FactPair>
LandmarkFactoryZhuGivanActions::apply_operator_and_propagate_labels(
    const OperatorProxy &op, const PropLayer &current, PropLayer &next) const {
    assert(operator_is_applicable(op, current));
    int op_id = get_operator_or_axiom_id(op);
    OpLabelSet precondition_labels =
        union_of_condition_labels(op.get_preconditions(), current);

    utils::HashSet<FactPair> changed;
    for (const EffectProxy &effect : op.get_effects()) {
        FactPair atom = effect.get_fact().get_pair();
        PropNode &next_node = next[atom.var][atom.value];
        if (next_node.reached && next_node.labels.empty()) {
            // Minimal already (no action landmark); intersection cannot shrink.
            continue;
        }
        if (conditions_hold(effect.get_conditions(), current)) {
            OpLabelSet contribution =
                union_of_condition_labels(effect.get_conditions(), current);
            contribution.insert(
                precondition_labels.begin(), precondition_labels.end());
            // The operator itself is a landmark for everything it reaches.
            contribution.insert(op_id);
            if (merge_action_labels(next_node, contribution)) {
                changed.insert(atom);
            }
        }
    }
    return changed;
}

LandmarkFactoryZhuGivanActions::PropLayer
LandmarkFactoryZhuGivanActions::initialize_layer(
    const TaskProxy &task_proxy, unordered_set<int> &triggered_ops) const {
    const State &initial_state = task_proxy.get_initial_state();
    const VariablesProxy &variables = task_proxy.get_variables();
    PropLayer layer;
    layer.resize(variables.size());
    for (VariableProxy var : variables) {
        int var_id = var.get_id();
        layer[var_id].resize(var.get_domain_size());
        // Initial-state facts are reached with no required action (empty labels).
        int value = initial_state[var].get_value();
        layer[var_id][value].reached = true;
        triggered_ops.insert(
            triggers[var_id][value].begin(), triggers[var_id][value].end());
    }
    return layer;
}

void LandmarkFactoryZhuGivanActions::propagate_until_fixpoint(
    const TaskProxy &task_proxy, unordered_set<int> &&triggered_ops,
    PropLayer &current) const {
    bool changes = true;
    while (changes) {
        PropLayer next(current);
        unordered_set<int> next_triggers;
        changes = false;
        for (int op_or_axiom_id : triggered_ops) {
            const OperatorProxy &op =
                get_operator_or_axiom(task_proxy, op_or_axiom_id);
            if (operator_is_applicable(op, current)) {
                utils::HashSet<FactPair> changed =
                    apply_operator_and_propagate_labels(op, current, next);
                if (!changed.empty()) {
                    changes = true;
                    for (const FactPair &atom : changed) {
                        next_triggers.insert(
                            triggers[atom.var][atom.value].begin(),
                            triggers[atom.var][atom.value].end());
                    }
                }
            }
        }
        swap(current, next);
        swap(triggered_ops, next_triggers);
    }
}

LandmarkFactoryZhuGivanActions::PropLayer
LandmarkFactoryZhuGivanActions::build_labeled_graph(
    const TaskProxy &task_proxy) const {
    assert(!triggers.empty());
    unordered_set<int> triggered_ops(
        task_proxy.get_operators().size() + task_proxy.get_axioms().size());
    PropLayer current = initialize_layer(task_proxy, triggered_ops);
    /* Operators without preconditions never propagate labels from a changing
       proposition, so (absent conditional effects) it suffices to apply them
       once at the start. */
    triggered_ops.insert(
        operators_without_preconditions.begin(),
        operators_without_preconditions.end());
    propagate_until_fixpoint(task_proxy, move(triggered_ops), current);
    return current;
}

// ── extraction and output ─────────────────────────────────────────────────────
LandmarkFactoryZhuGivanActions::OpLabelSet
LandmarkFactoryZhuGivanActions::collect_goal_action_landmarks(
    const TaskProxy &task_proxy, const PropLayer &layer) const {
    OpLabelSet result;
    for (FactProxy goal : task_proxy.get_goals()) {
        auto [var, value] = goal.get_pair();
        const PropNode &node = layer[var][value];
        if (!node.reached && log.is_at_least_normal()) {
            log << "Goal atom " << goal.get_name()
                << " is unreachable even in the delete relaxation; the action "
                << "landmarks reported below are incomplete." << endl;
            continue;
        }
        result.insert(node.labels.begin(), node.labels.end());
    }
    return result;
}

/* The label set of the AND node for `op`: {op} union the labels of its
   preconditions, read off the fixpoint layer. */
LandmarkFactoryZhuGivanActions::OpLabelSet
LandmarkFactoryZhuGivanActions::action_label_set_of_operator(
    const OperatorProxy &op, const PropLayer &layer) const {
    OpLabelSet labels = union_of_condition_labels(op.get_preconditions(), layer);
    labels.insert(get_operator_or_axiom_id(op));
    return labels;
}

static string json_escape(const string &s) {
    string out;
    out.reserve(s.size());
    for (char c : s) {
        if (c == '"' || c == '\\') {
            out.push_back('\\');
        }
        out.push_back(c);
    }
    return out;
}

void LandmarkFactoryZhuGivanActions::dump_action_landmarks(
    const TaskProxy &task_proxy, const PropLayer &layer,
    const OpLabelSet &goal_action_landmarks) const {
    // Deterministic, name-sorted output for easy parsing/diffing.
    vector<int> ids(
        goal_action_landmarks.begin(), goal_action_landmarks.end());
    sort(ids.begin(), ids.end(), [&](int a, int b) {
        return get_operator_or_axiom(task_proxy, a).get_name() <
               get_operator_or_axiom(task_proxy, b).get_name();
    });

    // Single-line JSON between markers so it is trivial to extract from FD's
    // mixed stdout and load with any JSON parser.
    cout << "<action-landmarks-json>\n";
    cout << "{\"num_action_landmarks\": " << ids.size()
         << ", \"action_landmarks\": [";
    for (size_t i = 0; i < ids.size(); ++i) {
        cout << (i ? ", " : "") << '"'
             << json_escape(get_operator_or_axiom(task_proxy, ids[i]).get_name())
             << '"';
    }
    cout << "], \"dependencies\": {";
    bool first = true;
    for (int op_id : ids) {
        OperatorProxy op = get_operator_or_axiom(task_proxy, op_id);
        // Required predecessor action landmarks: the action labels of this
        // operator's AND node that are themselves goal action landmarks.
        OpLabelSet label_set = action_label_set_of_operator(op, layer);
        vector<string> requirements;
        for (int dep_id : label_set) {
            if (dep_id != op_id && goal_action_landmarks.count(dep_id)) {
                requirements.push_back(
                    get_operator_or_axiom(task_proxy, dep_id).get_name());
            }
        }
        sort(requirements.begin(), requirements.end());
        cout << (first ? "" : ", ") << '"' << json_escape(op.get_name())
             << "\": [";
        for (size_t i = 0; i < requirements.size(); ++i) {
            cout << (i ? ", " : "") << '"' << json_escape(requirements[i])
                 << '"';
        }
        cout << "]";
        first = false;
    }
    cout << "}}\n";
    cout << "</action-landmarks-json>" << endl;
}

void LandmarkFactoryZhuGivanActions::add_goal_atoms_as_landmarks(
    const TaskProxy &task_proxy) const {
    for (FactProxy goal : task_proxy.get_goals()) {
        FactPair atom = goal.get_pair();
        if (!landmark_graph->contains_atomic_landmark(atom)) {
            Landmark landmark({atom}, ATOMIC, true);
            landmark_graph->add_landmark(move(landmark));
        } else {
            landmark_graph->get_atomic_landmark_node(atom)
                .get_landmark()
                .is_true_in_goal = true;
        }
    }
}

bool LandmarkFactoryZhuGivanActions::supports_conditional_effects() const {
    return true;
}

class LandmarkFactoryZhuGivanActionsFeature
    : public plugins::TypedFeature<
          LandmarkFactory, LandmarkFactoryZhuGivanActions> {
public:
    LandmarkFactoryZhuGivanActionsFeature() : TypedFeature("lm_zg_action") {
        document_title("Zhu/Givan action landmarks");
        document_synopsis(
            "Computes ACTION landmarks via Zhu/Givan label propagation and "
            "dumps them to stdout inside an <action-landmarks> block. This is a "
            "self-contained companion to lm_zg (the fact-based factory), which "
            "it does not modify. The landmark graph itself is populated only "
            "with the goal atoms, so any heuristic using this factory stays "
            "sound.");
        add_landmark_factory_options_to_feature(*this);
        document_language_support(
            "conditional_effects",
            "Conditional effects are handled during propagation; the printed "
            "predecessor dependencies assume operators with plain "
            "preconditions.");
    }

    virtual shared_ptr<LandmarkFactoryZhuGivanActions> create_component(
        const plugins::Options &opts) const override {
        return plugins::make_shared_from_arg_tuples<
            LandmarkFactoryZhuGivanActions>(
            get_landmark_factory_arguments_from_options(opts));
    }
};

static plugins::FeaturePlugin<LandmarkFactoryZhuGivanActionsFeature> _plugin;
}
