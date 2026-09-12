using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.CSharp.Syntax;

if (args.Length != 2)
    throw new ArgumentException("Usage: EnemyReference <decompiled directory> <output.json>");
var root = Path.GetFullPath(args[0]);
var hashes = new SortedDictionary<string, string>();
var common = new HashSet<string> { "StrengthPower", "DexterityPower", "WeakPower", "VulnerablePower", "FrailPower", "ArtifactPower" };
string ReadMechanics(string file)
{
    var source = File.ReadAllText(file);
    hashes[Path.GetRelativePath(root, file).Replace('\\', '/')] = Convert.ToHexString(SHA256.HashData(Encoding.UTF8.GetBytes(source)));
    var syntax = CSharpSyntaxTree.ParseText(source);
    if (syntax.GetDiagnostics().Any(d => d.Severity == DiagnosticSeverity.Error))
        throw new InvalidDataException(file);
    var type = syntax.GetRoot().DescendantNodes().OfType<ClassDeclarationSyntax>().First();
    // Preserve complete gameplay method bodies and helper definitions. Filtering
    // whole presentation members cannot change branch weights or conditions.
    var members = type.Members.Where(member => member switch {
        MethodDeclarationSyntax m => !new[] { "GenerateAnimator", "MonsterMoveList" }.Contains(m.Identifier.Text),
        PropertyDeclarationSyntax p => !Regex.IsMatch(p.Identifier.Text, "Sfx|Vfx|Anim|Visual|Asset|Texture|Scene|ShouldFade|ShouldDisappear"),
        _ => true
    }).Select(m => (MemberDeclarationSyntax)new PresentationFilter().Visit(m)!).ToList();
    // Remove now-unreferenced private presentation helpers, retaining overrides
    // and every dependency used by gameplay conditions or effect calculations.
    bool changed;
    do {
        changed = false;
        foreach (var member in members.ToArray()) {
            if (!member.Modifiers.Any(SyntaxKind.PrivateKeyword)) continue;
            var names = member switch {
                MethodDeclarationSyntax m => new[] { m.Identifier.Text },
                PropertyDeclarationSyntax p => new[] { p.Identifier.Text },
                FieldDeclarationSyntax f => f.Declaration.Variables.Select(v => v.Identifier.Text).ToArray(),
                _ => Array.Empty<string>()
            };
            if (names.Length == 0) continue;
            if (!members.Where(m => m != member).SelectMany(m => m.DescendantTokens())
                .Any(t => t.IsKind(SyntaxKind.IdentifierToken) && names.Contains(t.Text))) {
                members.Remove(member);
                changed = true;
            }
        }
    } while (changed);
    return string.Join("\n", members.Select(m => m.NormalizeWhitespace(eol: "\n").ToFullString()));
}
var monsters = new SortedDictionary<string, object>();
foreach (var file in Directory.GetFiles(Path.Combine(root, "MegaCrit.Sts2.Core.Models.Monsters"), "*.cs").Order())
{
    var name = Path.GetFileNameWithoutExtension(file);
    var id = Regex.Replace(Regex.Replace(name, "([A-Z]+)([A-Z][a-z])", "$1_$2"), "([a-z0-9])([A-Z])", "$1_$2").ToUpperInvariant();
    var mechanics = ReadMechanics(file);
    var powers = new SortedDictionary<string, string>();
    foreach (var power in Regex.Matches(mechanics, @"\b[A-Z]\w*Power\b").Select(m => m.Value).Distinct())
    {
        var powerFile = Path.Combine(root, "MegaCrit.Sts2.Core.Models.Powers", power + ".cs");
        if (!common.Contains(power) && File.Exists(powerFile)) powers[power] = ReadMechanics(powerFile);
    }
    monsters[id] = new { name, mechanics, powers };
}
var engine = ReadMechanics(Path.Combine(root, "MegaCrit.Sts2.Core.MonsterMoves.MonsterMoveStateMachine", "RandomBranchState.cs"));
var output = new { schema_version = 1, source = "Local decompiled build; static definitions only, never live AI state", source_sha256 = hashes, random_branch_semantics = engine, monsters };
File.WriteAllText(args[1], JsonSerializer.Serialize(output, new JsonSerializerOptions { WriteIndented = true }));
Console.WriteLine($"Generated {monsters.Count} monster references from {hashes.Count} source files.");

sealed class PresentationFilter : CSharpSyntaxRewriter
{
    public override SyntaxNode? VisitExpressionStatement(ExpressionStatementSyntax node)
    {
        var text = node.Expression.ToString();
        // Only standalone presentation calls are removable. Never rewrite
        // conditions, random weights, damage chains or gameplay command args.
        if (Regex.IsMatch(text, @"^(await\s+)?(SfxCmd\.|VfxCmd\.|CreatureCmd\.TriggerAnim\(|Cmd\.(Wait|CustomScaledWait)\(|NGame\.Instance\?\.ScreenShake\(|NCombatRoom\.Instance\?\.(RadialBlur|DoHitStop|CombatVfxContainer)\b)"))
            return null;
        return base.VisitExpressionStatement(node);
    }
}
