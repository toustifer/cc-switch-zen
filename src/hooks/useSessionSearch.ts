import { useCallback, useMemo } from "react";
import type { SessionMeta } from "@/types";

interface UseSessionSearchOptions {
  sessions: SessionMeta[];
  providerFilter: string;
}

interface UseSessionSearchResult {
  search: (query: string) => SessionMeta[];
}

interface SearchableSession {
  session: SessionMeta;
  haystack: string;
}

export function useSessionSearch({
  sessions,
  providerFilter,
}: UseSessionSearchOptions): UseSessionSearchResult {
  const filteredByProvider = useMemo(() => {
    if (providerFilter === "all") return sessions;
    return sessions.filter((s) => s.providerId === providerFilter);
  }, [sessions, providerFilter]);

  const searchableSessions = useMemo<SearchableSession[]>(() => {
    return filteredByProvider.map((session) => ({
      session,
      haystack: [
        session.sessionId,
        session.title,
        session.summary,
        session.projectDir,
        session.sourcePath,
      ]
        .filter(Boolean)
        .join(" ")
        .toLowerCase(),
    }));
  }, [filteredByProvider]);

  const search = useCallback(
    (query: string): SessionMeta[] => {
      const needle = query.trim().toLowerCase();

      if (!needle) {
        return [...filteredByProvider].sort((a, b) => {
          const aTs = a.lastActiveAt ?? a.createdAt ?? 0;
          const bTs = b.lastActiveAt ?? b.createdAt ?? 0;
          return bTs - aTs;
        });
      }

      return searchableSessions
        .map(({ session, haystack }) => ({
          session,
          matchIndex: haystack.indexOf(needle),
        }))
        .filter(({ matchIndex }) => matchIndex !== -1)
        .sort((a, b) => {
          if (a.matchIndex !== b.matchIndex) {
            return a.matchIndex - b.matchIndex;
          }

          const aTs = a.session.lastActiveAt ?? a.session.createdAt ?? 0;
          const bTs = b.session.lastActiveAt ?? b.session.createdAt ?? 0;
          return bTs - aTs;
        })
        .map(({ session }) => session);
    },
    [filteredByProvider, searchableSessions],
  );

  return { search };
}
