def get_tier(matrix, category, tier):
    try:
        return matrix[category]['TIERS'][tier]
    except Exception:
        print(f'could not get tier for category: {category}, tier: {tier}')
        return []