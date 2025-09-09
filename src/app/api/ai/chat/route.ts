import { NextRequest, NextResponse } from "next/server";
import { GoogleGenerativeAI } from "@google/generative-ai";

const MODEL = "gemini-1.5-flash";

const system = `You are a specialized assistant ONLY for the CabShare app at NIT Rourkela campus.

STRICT RULES:
- ONLY answer questions about cabsharing, ride posting, ride finding, campus transportation, safety tips, and pricing for rides
- If asked about anything else (weather, sports, academics, general topics, etc.), politely decline and redirect to cabshare topics
- Always stay focused on helping with ride-related queries
- Be brief, helpful, and campus-specific

TOPICS YOU CAN HELP WITH:
- How to post a ride
- How to find rides
- Safety tips for sharing rides
- Fair pricing and cost splitting
- Contacting other riders/drivers
- Campus-specific transportation advice
- Distance and travel time between locations (I can calculate this for you)

DISTANCE CALCULATION:
When users ask about distance between two locations, respond with: "I can calculate the distance and travel time for you. Let me check..." and then use the CALCULATE_DISTANCE function.

If the question is not about cabsharing or campus rides, respond: "I'm specifically designed to help with CabShare app questions. Please ask me about posting rides, finding rides, safety tips, or pricing for campus transportation."`;

type ChatMessage = { role: "user" | "assistant" | "system"; content: string };

// --- Utilities --------------------------------------------------------------

function sanitizePlace(raw: string): string {
  return raw
    .replace(/\s{2,}/g, " ")
    .trim()
    .replace(/^[\s,.;:]+|[\s,.;:]+$/g, ""); // only strip trivial punctuation
}

function maybeExpandCampusAlias(name: string): string {
  const lower = name.toLowerCase();
  const campusish = /(main gate|front gate|back gate|sac|hostel|lecture avenue|tiir|tsg|dtp|sac building)/i;
  // If user mentions a campus alias but not NIT/Rourkela, append campus context
  if (campusish.test(lower) && !/nit|rourkela/.test(lower)) {
    return `${name}, NIT Rourkela, Odisha`;
  }
  return name;
}

// Robustly extract two locations from free text
function extractLocations(text: string): { origin: string; destination: string } | null {
  const t = text.trim();

  // Removed unused tryMatch function

  const patterns: RegExp[] = [
    /\bfrom\s+(.+?)\s+(?:to|→|and)\s+(.+?)(?=[.?!,;]|$)/i,
    /\bbetween\s+(.+?)\s+(?:and|to)\s+(.+?)(?=[.?!,;]|$)/i,
    /\bhow\s+far(?:\s+is\s+it)?\s+(.+?)\s+(?:to|from)\s+(.+?)(?=[.?!,;]|$)/i,
    /(.+?)\s+(?:to|→|and)\s+(.+?)(?=[.?!,;]|$)/i,
  ];

  for (const re of patterns) {
    const m = t.match(re);
    if (m) {
      const a = sanitizePlace(m[1]);
      const b = sanitizePlace(m[2]);
      if (a && b && a.toLowerCase() !== b.toLowerCase()) {
        return { origin: a, destination: b };
      }
    }
  }
    // ✅ Fallback: handles "X from Y"
    const m2 = t.match(/(.+?)\s+from\s+(.+?)(?=[.?!,;]|$)/i);
    if (m2) {
      return {
        origin: sanitizePlace(m2[2]), // the part after "from"
        destination: sanitizePlace(m2[1]), // the part before "from"
      };
    }
  return null;
}

async function geocodeToPlaceId(q: string, mapsKey: string): Promise<{ placeId: string; formatted: string } | null> {
  try {
    const url = new URL("https://maps.googleapis.com/maps/api/geocode/json");
    url.searchParams.set("address", q);
    url.searchParams.set("key", mapsKey);
    // Bias to India (campus context)
    url.searchParams.set("components", "country:IN");
    url.searchParams.set("region", "in");
    url.searchParams.set("language", "en");
    const res = await fetch(url.toString());
    const data = await res.json();
    if (data.status !== "OK" || !data.results?.length) return null;
    const r = data.results[0];
    return { placeId: r.place_id, formatted: r.formatted_address };
  } catch {
    return null;
  }
}

async function calculateDistance(originRaw: string, destinationRaw: string) {
  try {
    const mapsKey = process.env.GOOGLE_MAPS_API_KEY;
    if (!mapsKey) {
      console.log("Missing GOOGLE_MAPS_API_KEY");
      return null;
    }

    // First geocode both ends to stabilize the query
    const [o, d] = await Promise.all([
      geocodeToPlaceId(originRaw, mapsKey),
      geocodeToPlaceId(destinationRaw, mapsKey),
    ]);

    // Fall back to raw strings if geocoding fails
    const originParam = o ? `place_id:${o.placeId}` : originRaw;
    const destParam = d ? `place_id:${d.placeId}` : destinationRaw;

    const url = new URL("https://maps.googleapis.com/maps/api/directions/json");
    url.searchParams.set("origin", originParam);
    url.searchParams.set("destination", destParam);
    url.searchParams.set("key", mapsKey);
    url.searchParams.set("alternatives", "false");
    url.searchParams.set("departure_time", "now");
    url.searchParams.set("mode", "driving");
    url.searchParams.set("units", "metric");
    url.searchParams.set("region", "in");

    console.log("Fetching distance for:", originParam, "to", destParam);
    const res = await fetch(url.toString());
    const data = await res.json();

    console.log("Google Maps API response status:", data.status);

    if (data.status !== "OK" || !data.routes?.length) {
      console.log("API Error:", data.error_message || data.status);
      return null;
    }

    const leg = data.routes[0].legs[0];
    const distanceKm = (leg.distance.value / 1000).toFixed(1);
    const durationMins = Math.round((leg.duration_in_traffic?.value ?? leg.duration.value) / 60);

    const originLabel = o?.formatted ?? leg.start_address;
    const destLabel = d?.formatted ?? leg.end_address;

    console.log("Distance calculated:", distanceKm, "km,", durationMins, "mins");
    return {
      distanceKm,
      durationMins,
      originLabel,
      destLabel,
    };
  } catch (error) {
    console.error("Distance calculation error:", error);
    return null;
  }
}

// --- Route handler ----------------------------------------------------------

export async function POST(req: NextRequest) {
  try {
    const { messages } = (await req.json()) as { messages: ChatMessage[] };
    if (!Array.isArray(messages)) {
      return NextResponse.json({ error: "messages array required" }, { status: 400 });
    }

    const key = process.env.GEMINI_API_KEY;
    if (!key) return NextResponse.json({ error: "Missing GEMINI_API_KEY" }, { status: 500 });

    // Check for distance intent (wider net)
    const latest = messages[messages.length - 1]?.content ?? "";
    const userWindow = messages
      .filter((m) => m.role === "user")
      .slice(-5) // look across the last few user turns
      .map((m) => m.content)
      .join(" \n ");

    const distanceIntent = /\b(distance|how\s+far|travel\s*time|route|directions|eta|reach|kilomet(?:er|re)s?|km|mins?|minutes?)\b/i;

    const textToParse = distanceIntent.test(latest) ? latest : userWindow;

    let parsed = null as null | { origin: string; destination: string };
    if (distanceIntent.test(textToParse)) {
      parsed = extractLocations(textToParse);
      // If not found in latest/userWindow, try a lenient two-quoted fallback
      if (!parsed) {
        const qm = textToParse.match(/["“'‘]([^"“”'’]+)["”'’][\s,;:]+["“'‘]([^"“”'’]+)["”'’]/u);
        if (qm) {
          parsed = {
            origin: maybeExpandCampusAlias(sanitizePlace(qm[1])),
            destination: maybeExpandCampusAlias(sanitizePlace(qm[2])),
          };
        }
      }
    }

    if (parsed) {
      const { origin, destination } = parsed;

      const info = await calculateDistance(origin, destination);

      if (info) {
        const reply =
          `I can calculate the distance and travel time for you. Let me check...\n\n` +
          `**From:** ${info.originLabel}\n` +
          `**To:** ${info.destLabel}\n` +
          `**Distance:** ${info.distanceKm} km\n` +
          `**Travel time:** ${info.durationMins} minutes\n\n` +
          `Tip: Use this to split cab costs fairly among passengers.`;
        return NextResponse.json({ reply });
      } else {
        const reply =
          `I tried to calculate the distance but couldn't resolve one of the places.\n` +
          `Please try again with clearer names (e.g., "NIT Rourkela Main Gate to Rourkela Railway Station").`;
        return NextResponse.json({ reply });
      }
    }

    // Fallback to normal Gemini reply under the strict system
    const genAI = new GoogleGenerativeAI(key);
    const model = genAI.getGenerativeModel({ model: MODEL });
    const convo = messages.map((m) => `${m.role}: ${m.content}`).join("\n");
    const prompt = `${system}\n\nConversation so far:\n${convo}\n\nReply:`;
    const result = await model.generateContent([{ text: prompt }]);
    const reply = result.response.text();
    return NextResponse.json({ reply });
  } catch (e) {
    const msg = e instanceof Error ? e.message : "chat error";
    return NextResponse.json({ error: msg }, { status: 500 });
  }
}
