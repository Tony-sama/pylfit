# Existence: event appears at least once
def existence(events, sequence):
    features = []
    values = []
    for e in sorted(events):
        features.append("existence_"+str(e))
        values.append(e in sequence)
    return features, values

# Absence 2: event appears at most once
def absence_2(events, sequence):
    features = []
    values = []
    for e in sorted(events):
        features.append("absence_2_"+str(e))
        values.append(sequence.count(e) < 2)
    return features, values

# Choice: event appears at most once
def choice(events, sequence):
    features = []
    values = []
    for i, ei in enumerate(sorted(events)):
        for ej in sorted(events)[i+1:]:
            features.append("choice_"+str(ei)+"_"+str(ej))
            values.append(ei in sequence or ej in sequence)
    return features, values

# Exclusive choice: event appears at most once
def exclusive_choice(events, sequence):
    features = []
    values = []
    for i, ei in enumerate(sorted(events)):
        for ej in sorted(events)[i+1:]:
            features.append("exclusive_choice_"+str(ei)+"_"+str(ej))
            values.append((ei in sequence and ej not in sequence) or (ei not in sequence and ej in sequence))
    return features, values

# Resp. existence: if a appears b must appears
def resp_existence(events, sequence):
    features = []
    values = []
    for i, ei in enumerate(sorted(events)):
        for ej in sorted(events)[i+1:]:
            features.append("resp_existence_"+str(ei)+"_"+str(ej))
            values.append((ei in sequence and ej in sequence) or (ei not in sequence))
    return features, values

# Coexistence: a and b must appears or none of them
def coexistence(events, sequence):
    features = []
    values = []
    for i, ei in enumerate(sorted(events)):
        for ej in sorted(events)[i+1:]:
            features.append("coexistence_"+str(ei)+"_"+str(ej))
            values.append((ei in sequence and ej in sequence) or (ei not in sequence and ej not in sequence))
    return features, values

# Response: every time a appears b must appear in the future
def response(events, sequence):
    features = []
    values = []
    for ei in sorted(events):
        for ej in sorted(events):
            if(ei == ej):
                continue
            features.append("response_"+str(ei)+"_"+str(ej))
            ei_indices = [i for i, x in enumerate(sequence) if x == ei] + [-1]
            ej_indices = [i for i, x in enumerate(sequence) if x == ej] + [-1]

            last_ei = max(ei_indices)
            last_ej = max(ej_indices)

            values.append(last_ei <= last_ej)
    return features, values

# Precedence: b can be executed if a has been executed before
def precedence(events, sequence):
    features = []
    values = []
    for ei in sorted(events):
        for ej in sorted(events):
            if(ei == ej):
                continue
            features.append("precedence_"+str(ei)+"_"+str(ej))
            ei_indices = [i for i, x in enumerate(sequence) if x == ei] + [-1]
            ej_indices = [i for i, x in enumerate(sequence) if x == ej] + [-1]

            first_ei = min(ei_indices)
            first_ej = min(ej_indices)

            if(ej not in sequence):
                values.append(True)
            elif (ei not in sequence):
                values.append(False)
            else:
                values.append(first_ei < first_ej)
    return features, values

# Succession: b must be executed after a and a must precede b
def succession(events, sequence):
    features = []
    values = []
    for ei in sorted(events):
        for ej in sorted(events):
            if(ei == ej):
                continue
            features.append("succession_"+str(ei)+"_"+str(ej))
            ei_indices = [i for i, x in enumerate(sequence) if x == ei] + [-1]
            ej_indices = [i for i, x in enumerate(sequence) if x == ej] + [-1]

            last_ei = max(ei_indices)
            last_ej = max(ej_indices)

            a_before_b = (ej not in sequence) or (last_ej > last_ei)
            response_a_b = last_ei <= last_ej
            values.append(response_a_b  and a_before_b)
    return features, values

# Alternate

# Alt. Response: every a must be followed by b, without any other a inbetween
def alt_response(events, sequence):
    features = []
    values = []
    for ei in sorted(events):
        for ej in sorted(events):
            if(ei == ej):
                continue
            features.append("alt_response_"+str(ei)+"_"+str(ej))

            value = True
            expect_ej = False
            for e in sequence:
                if(expect_ej and e == ei):
                    value = False
                    break

                if(e == ei):
                    expect_ej = True
                    continue

                if(e == ej):
                    expect_ej = False

            if(expect_ej):
                value = False

            values.append(value)
    return features, values

# Alt. Precedence: every b must be prededed by a without any other b in between
def alt_precedence(events, sequence):
    features = []
    values = []
    for ei in sorted(events):
        for ej in sorted(events):
            if(ei == ej):
                continue
            features.append("alt_precedence_"+str(ei)+"_"+str(ej))
            value = True
            expect_ei = False
            for e in reversed(sequence):
                if(expect_ei and e == ej):
                    value = False
                    break

                if(e == ej):
                    expect_ei = True
                    continue

                if(e == ei):
                    expect_ei = False

            if(expect_ei):
                value = False

            values.append(value)
    return features, values

# Alt. Succession: combination of alternate response and alternate precedence
def alt_succession(events, sequence):
    features = []
    values = []
    for ei in sorted(events):
        for ej in sorted(events):
            if(ei == ej):
                continue
            features.append("alt_succession_"+str(ei)+"_"+str(ej))

    values = [a and b for a, b in zip(alt_response(events, sequence)[1], alt_precedence(events, sequence)[1])]
    return features, values

# Chains

# Chain Response: if a appears then b must appears immediatly after
def chain_response(events, sequence):
    features = []
    values = []
    for ei in sorted(events):
        for ej in sorted(events):
            if(ei == ej):
                continue
            features.append("chain_response_"+str(ei)+"_"+str(ej))

            value = True
            for i, e in enumerate(sequence):
                if(e == ei and (i >= len(sequence)-1 or sequence[i+1] != ej)):
                    value = False
                    break
            values.append(value)
    return features, values

# Chain Precedence: b can appear only immediatly after a
def chain_precedence(events, sequence):
    features = []
    values = []
    for ei in sorted(events):
        for ej in sorted(events):
            if(ei == ej):
                continue
            features.append("chain_precedence_"+str(ei)+"_"+str(ej))

            value = True
            r_sequence = sequence.copy()
            r_sequence.reverse()
            for i, e in enumerate(r_sequence):
                if(e == ej and (i >= len(r_sequence)-1 or r_sequence[i+1] != ei)):
                    value = False
                    break
            values.append(value)
    return features, values

# Chain Succession: a and b must appear next to each other
def chain_succession(events, sequence):
    features = []
    values = []
    for ei in sorted(events):
        for ej in sorted(events):
            if(ei == ej):
                continue
            features.append("chain_succession_"+str(ei)+"_"+str(ej))

    values = [a and b for a, b in zip(chain_response(events, sequence)[1], chain_precedence(events, sequence)[1])]
    return features, values

# Negations

# Not Coexistence: only one among a and b can appears but not both
def not_coexistence(events, sequence):
    features = []
    values = []
    for idx, ei in enumerate(sorted(events)):
        for ej in sorted(events)[idx+1:]:
            if(ei == ej):
                continue
            features.append("not_coexistence_"+str(ei)+"_"+str(ej))

            values.append(not (ei in sequence and ei in sequence))
    return features, values

# Not Succession: a cannot be followed by b and b cannot be precede by a
def not_succession(events, sequence):
    features = []
    values = []
    for ei in sorted(events):
        for ej in sorted(events):
            if(ei == ej):
                continue
            features.append("not_succession_"+str(ei)+"_"+str(ej))

            ei_indices = [i for i, x in enumerate(sequence) if x == ei]
            ej_indices = [i for i, x in enumerate(sequence) if x == ej]

            value = True

            if(ei in sequence and ej in sequence):
                first_ei = min(ei_indices)
                last_ej = max(ej_indices)
                value = first_ei > last_ej

            values.append(value)
    return features, values

# Not Chain Succession: a and b cannot appears next to each other
def not_chain_succession(events, sequence):
    features = []
    values = []
    for idx, ei in enumerate(sorted(events)):
        for ej in sorted(events)[idx+1:]:
            if(ei == ej):
                continue
            features.append("not_chain_succession_"+str(ei)+"_"+str(ej))

            value = True
            for i, e in enumerate(sequence):
                if(i >= len(sequence)-1):
                    break
                if(e == ei and sequence[i+1] == ej or e == ej and sequence[i+1] == ei):
                    value = False
                    break
            values.append(value)
    return features, values


# Not Precedence: there is no a before any b
def not_precedence(events, sequence):
    features = []
    values = []
    for ei in sorted(events):
        for ej in sorted(events):
            if(ei == ej):
                continue
            features.append("not_precedence_"+str(ei)+"_"+str(ej))
            ei_indices = [i for i, x in enumerate(sequence) if x == ei]
            ej_indices = [i for i, x in enumerate(sequence) if x == ej]

            if(ej not in sequence or ei not in sequence):
                values.append(True)
            else:
                first_ei = min(ei_indices)
                last_ej = max(ej_indices)

                values.append(first_ei > last_ej)
    return features, values
